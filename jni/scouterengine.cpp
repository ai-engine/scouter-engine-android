#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include <android/log.h>


#define LOG_TAG "JNI_PART"
#define LOGV(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace std;

//#define _DEBUG
const int gUseRatioTest = false;

class PatternDetector {

private:
	int mImageNum;

	cv::OrbFeatureDetector* mDetector;
	cv::OrbFeatureDetector* mDetectorCenter;
	cv::OrbDescriptorExtractor* mExtractor;
	cv::FlannBasedMatcher* mMatcher;

	vector< vector<cv::KeyPoint> > mTrainKeypoints;
	bool mFlgHoldTrainKeypoints;

	const string FILENAME_IMAGE_FEATURES;

public:
	PatternDetector():
		FILENAME_IMAGE_FEATURES("/sdcard/imageFeatures.dat")
	{
		mDetector = new cv::OrbFeatureDetector(2000);

		mExtractor = new cv::OrbDescriptorExtractor();
		mMatcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(16,14,0));
		mFlgHoldTrainKeypoints = true;
		clear();
	}

	void clear() {
		mImageNum = 0;
		mTrainKeypoints.clear();
	}

	void addTrainingImage(cv::Mat& trainImage) {
		vector<cv::KeyPoint> trainKeypoints;
		cv::Mat trainDescriptors;

		mDetector->detect(trainImage, trainKeypoints);
		mExtractor->compute(trainImage, trainKeypoints, trainDescriptors);

		vector<cv::Mat> descriptors(1);
		descriptors[0] = trainDescriptors.clone();
		mMatcher->add(descriptors);
		mImageNum++;

		LOGV(":rm im%d.jpg #%d", mImageNum, trainKeypoints.size());

		if (mFlgHoldTrainKeypoints) {
			mTrainKeypoints.push_back(trainKeypoints);
		}
	}

	void drawKeypoints(cv::Mat& image, vector<cv::KeyPoint>& keypoints) {
		for( size_t i = 0; i < keypoints.size(); i++ ) {
			cv::circle(image, cv::Point(keypoints[i].pt.x, keypoints[i].pt.y), 3, cv::Scalar(0,0,255,255));
		}
	}

	int getBestMatch(const cv::Mat& queryDescriptors, vector<cv::DMatch>* matches = NULL)
	{
		vector< vector<cv::DMatch> > knnMatches;

		const float minRatio = 1.f / 1.5f;
		mMatcher->knnMatch(queryDescriptors, knnMatches, 2);

		vector<int> votes(mImageNum);
		for (int i = 0; i < mImageNum; i++) {
			votes[i] = 0;
		}

		for (size_t i=0; i<knnMatches.size(); i++)
		{
			const cv::DMatch* bestMatch   = &knnMatches[i][0];
			const cv::DMatch* betterMatch = &knnMatches[i][1];

			if (bestMatch == NULL || betterMatch == NULL) {
				continue;
			}
			float distanceRatio = bestMatch->distance / (betterMatch->distance);

			if (distanceRatio < minRatio)
			{
				votes[bestMatch->imgIdx]++;
				if (matches != NULL) {
					matches->push_back(*bestMatch);
				}
			}
		}

		vector<cv::Mat> trainDescs = mMatcher->getTrainDescriptors();
		int maxImageId = -1;
		double maxVotes = 0;
		for (int i = 0; i < mImageNum; i++) {
			double match = (double)votes[i];

			if (match > maxVotes) {
				maxImageId = i;
				maxVotes = (double)votes[i];
			}
		}


		return maxImageId;
	}

	void createIndex() {
		mMatcher->train();
	}

	void write() {
		ofstream fout(FILENAME_IMAGE_FEATURES.c_str());
		fout << mMatcher->getTrainDescriptors()[0];
		fout.close();
	}
	void read() {
		cv::FileStorage fs(FILENAME_IMAGE_FEATURES, cv::FileStorage::READ);
		cv::FileNode fn = fs.getFirstTopLevelNode();
		mMatcher->clear();
		mMatcher->read(fn);
		fs.release();

		ofstream fin(FILENAME_IMAGE_FEATURES.c_str());
		fin.close();
	}

	int detectImage(cv::Mat& queryImage) {

		vector<cv::KeyPoint> queryKeypoints;
		cv::Mat queryDescriptors;
		mDetector->detect(queryImage, queryKeypoints);
		mExtractor->compute(queryImage, queryKeypoints, queryDescriptors);

		return getBestMatch(queryDescriptors);
	}

	int getPatternImagePositionWithHomography(cv::Mat& queryImage, double* ratioX, double* ratioY, double* size) {
		if (!mFlgHoldTrainKeypoints) return -1;

		vector<cv::KeyPoint> queryKeypoints;
		vector<cv::DMatch> matches;
		cv::Mat roughHomography;

		cv::Mat queryDescriptors;
		mDetector->detect(queryImage, queryKeypoints);
		mExtractor->compute(queryImage, queryKeypoints, queryDescriptors);

		int id = getBestMatch(queryDescriptors, &matches);

		if (id != -1) {
			getMatches(queryDescriptors, matches);
			int threthold = 1;
			int result = refineMatchesWithHomography(
					queryKeypoints,
					mTrainKeypoints[id],
					threthold,
					matches,
					roughHomography);

			LOGV("rough:%d", result);

			if (result) {
				stringstream sstr;
				sstr << roughHomography;

				vector<cv::Point2f> inPoints;
				vector<cv::Point2f> outPoints;
				for (int i=0; i<queryKeypoints.size(); i++) {
					inPoints.push_back(queryKeypoints[i].pt);
				}

				cv::Mat m_warpedImg;
				cv::warpPerspective(queryImage, m_warpedImg, roughHomography, cv::Size(queryImage.cols, queryImage.rows), cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);
				//cv::perspectiveTransform(inPoints, outPoints, roughHomography);

				std::vector<cv::KeyPoint> warpedKeypoints;
				std::vector<cv::DMatch> refinedMatches;

				// Detect features on warped image
				extractFeatures(m_warpedImg, warpedKeypoints, queryDescriptors);

				// Match with pattern
				getMatches(queryDescriptors, refinedMatches);

				// Estimate new refinement homography
				cv::Mat refinedHomography;
				int homographyFound = refineMatchesWithHomography(
					warpedKeypoints,
					queryKeypoints,
					threthold,
					refinedMatches,
					refinedHomography);

				LOGV("refine:%d", homographyFound);
				if (homographyFound) {
					// Get a result homography as result of matrix product of refined and rough homographies:
					cv::Mat homography = roughHomography * refinedHomography;

					// Transform contour with precise homography
					cv::perspectiveTransform(inPoints, outPoints,  homography);

					sstr << outPoints;
					LOGV("mat:%s", sstr.str().c_str());

					vector<float> xs;
					vector<float> ys;
					for (int i=0; i<queryKeypoints.size(); i++) {
						xs.push_back(outPoints[i].x);
						ys.push_back(outPoints[i].y);
					}
					sort(xs.begin(),xs.end());
					sort(ys.begin(),ys.end());
					*ratioY =  xs[xs.size()/2] / (double) queryImage.cols;
					*ratioX =  ys[ys.size()/2] / (double) queryImage.rows;
					//*size = xs[xs.size()*(2.5f/4.0f)]-xs[xs.size()*(1.5f/4.0f)] + ys[ys.size()*(2.5f/4.0f)]-ys[ys.size()*(1.5f/4.0f)];

					int idStart = xs.size()*(1.0f/4.0f);
					int idEnd   = xs.size()*(3.0f/4.0f);
					double meanx=0, meany=0;
					int count = 0;
					for (int i=idStart; i<idEnd; i++) {
						meanx += xs[i];
						meany += ys[i];
						count++;
					}
					meanx /= (double)count;
					meany /= (double)count;
					double sdx = 0, sdy = 0;
					for (int i=idStart; i<idEnd; i++) {
						sdx += (xs[i] - meanx)*(xs[i] - meanx);
						sdy += (ys[i] - meany)*(ys[i] - meany);
					}
					sdx /= (double)count;
					sdy /= (double)count;
					*size = sdx + sdy;

					LOGV("median:%f,%f", *ratioX,*ratioY);
					LOGV("25-75:%f", *size);
				}
			}
		}

		return id;
	}


	void getMatches(const cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches)
	{
	    matches.clear();
	    vector< vector<cv::DMatch> > m_knnMatches;

	    if (gUseRatioTest)
	    {
	        // To avoid NaN's when best match has zero distance we will use inversed ratio.
	        const float minRatio = 1.f / 1.5f;

	        // KNN match will return 2 nearest matches for each query descriptor
	        mMatcher->knnMatch(queryDescriptors, m_knnMatches, 2);

	        for (size_t i=0; i<m_knnMatches.size(); i++)
	        {
	            const cv::DMatch* bestMatch   = &m_knnMatches[i][0];
	            const cv::DMatch* betterMatch = &m_knnMatches[i][1];

	            if (bestMatch == NULL || betterMatch == NULL) {
					continue;
				}
	            float distanceRatio = bestMatch->distance / betterMatch->distance;

	            // Pass only matches where distance ratio between
	            // nearest matches is greater than 1.5 (distinct criteria)
	            if (distanceRatio < minRatio)
	            {
	                matches.push_back(*bestMatch);
	            }
	        }
	    }
	    else
	    {
	        // Perform regular match
	        mMatcher->match(queryDescriptors, matches);
	    }
	}

	bool extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
	{
	    assert(!image.empty());
	    assert(image.channels() == 1);

	    mDetector->detect(image, keypoints);
	    if (keypoints.empty())
	        return false;

	    mExtractor->compute(image, keypoints, descriptors);
	    if (keypoints.empty())
	        return false;

	    return true;
	}

	bool refineMatchesWithHomography
	    (
	    const std::vector<cv::KeyPoint>& queryKeypoints,
	    const std::vector<cv::KeyPoint>& trainKeypoints,
	    float reprojectionThreshold,
	    std::vector<cv::DMatch>& matches,
	    cv::Mat& homography
	    )
	{
	    const int minNumberMatchesAllowed = 8;

	    if (matches.size() < minNumberMatchesAllowed)
	        return false;

	    // Prepare data for cv::findHomography
	    std::vector<cv::Point2f> srcPoints(matches.size());
	    std::vector<cv::Point2f> dstPoints(matches.size());

	    for (size_t i = 0; i < matches.size(); i++)
	    {
	        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
	        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	    }

	    // Find homography matrix and get inliers mask
	    std::vector<unsigned char> inliersMask(srcPoints.size());
	    homography = cv::findHomography(srcPoints,
	                                    dstPoints,
	                                    CV_FM_RANSAC,
	                                    reprojectionThreshold,
	                                    inliersMask);

	    std::vector<cv::DMatch> inliers;
	    for (size_t i=0; i<inliersMask.size(); i++)
	    {
	        if (inliersMask[i])
	            inliers.push_back(matches[i]);
	    }

	    matches.swap(inliers);
	    return matches.size() > minNumberMatchesAllowed;
	}


};

extern "C" {

PatternDetector patternDetector;

JNIEXPORT void JNICALL Java_com_scouterengine_lib_SpecificObjectDetector_addTrainingImage(
		JNIEnv* env,
		jobject thiz,
		jint width,
		jint height,
		jintArray rgba
)
{
	jint* _rgba = env->GetIntArrayElements(rgba, 0);
	cv::Mat mrgba(height, width, CV_8UC4, (unsigned char *)_rgba);
	cv::Mat gray(height, width, CV_8UC1);
	cvtColor(mrgba, gray, CV_RGBA2GRAY, 0);

	patternDetector.addTrainingImage(gray);

	env->ReleaseIntArrayElements(rgba, _rgba, 0);
}

JNIEXPORT void JNICALL Java_com_scouterengine_lib_SpecificObjectDetector_createIndex(
		JNIEnv* env,
		jobject thiz
)
{
	patternDetector.createIndex();
}

JNIEXPORT void JNICALL Java_com_scouterengine_lib_SpecificObjectDetector_writeImageFeatures()
{
	patternDetector.write();
}

JNIEXPORT void JNICALL Java_com_scouterengine_lib_SpecificObjectDetector_readImageFeatures()
{
	patternDetector.read();
}

JNIEXPORT jint JNICALL Java_com_scouterengine_lib_SpecificObjectDetector_detectImage(
		JNIEnv* env,
		jobject thiz,
		jint width,
		jint height,
		jbyteArray yuv)
{
	jbyte* _yuv = env->GetByteArrayElements(yuv, 0);
	cv::Mat mgray(height, width, CV_8UC1, (unsigned char *) _yuv);
	int id = patternDetector.detectImage(mgray);
    env->ReleaseByteArrayElements(yuv, _yuv, 0);

    return id;
}



JNIEXPORT void JNICALL Java_com_scouterengine_lib_SpecificObjectDetector_clearImages(JNIEnv* env, jobject thiz)
{
	patternDetector.clear();
}


}
