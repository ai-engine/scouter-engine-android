package com.scouterengine.lib;


import android.graphics.Bitmap;


/**
 * specific object recognition
 * @author nakamura.masayoshi
 */
public class SpecificObjectDetector {
	
	private int mCntTemplateImages;
	final private int INVALID_ID = -1;
	
	public SpecificObjectDetector() {
		clearTemplateImages();
	}
	
	/**
	 * register template image
	 * @param bmp format image 
	 * @return image ID
	 */
	public int registerTemplateImage(Bitmap bmp) {
		try {
			int widths = bmp.getWidth();
			int heights = bmp.getHeight();
			int[] rgbas = new int[widths * heights];
			bmp.getPixels(rgbas, 0, widths, 0, 0, widths, heights);
			addTrainingImage(widths, heights, rgbas);
			bmp.recycle();
			return mCntTemplateImages++;
		} catch (Exception e) {
			e.printStackTrace();
			return INVALID_ID;
		}
	}
	
	/**
	 * return number of template images
	 * @return number
	 */
	public int getNumberOfTemplateImages() {
		return mCntTemplateImages;
	}

	/**
	 * unregister and clear template images
	 */
	public void clearTemplateImages() {
		clearImages();
		mCntTemplateImages = 0;
	}

	/**
	 * detect image
	 * @param data YCbCr_420_SP (NV21)
	 * @return detected image ID
	 */
	public int detect(int width, int height, byte[] data)
	{
		return detectImage(width, height, data);
	}


	/**
	 * create index
	 * call this function after registering all images by registerTemplateImage(...)
	 */
	public native void createIndex();
	
	
	public native void findFeatures(int width, int height, byte yuv[], int[] rgba);
	private native int detectImage(int width, int height, byte[] data);
	private native void clearImages();
	private native void finishTrainImage(int imageNum);
	private native void addTrainingImage(int width, int height, int[] rgbas);
	private native void writeImageFeatures();
	private native void readImageFeatures();
	
	static {
		System.loadLibrary("scouterengine");
		//System.loadLibrary("libscouterengine");
	}
	
}
