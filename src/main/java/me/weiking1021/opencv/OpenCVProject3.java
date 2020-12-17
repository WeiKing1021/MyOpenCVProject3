package me.weiking1021.opencv;

import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class OpenCVProject3 {

	static {

		Util.init();
	}

	public static final File ROOT = new File("").getAbsoluteFile();

	private static final int CANNY_THRESHOLD = 40;

	public static final double RHO = 2;
	public static final double THETA = 3 * Math.PI;
	public static final int THRESHOLD = 100;

	public static void main(String args[]) throws IOException {

		Mat mat = Util.image2cvMat(ImageIO.read(new File(ROOT + "./opencv/image.jpg")));

		Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);

		Imgproc.GaussianBlur(mat, mat, new Size(11, 11), 1);

		Mat show_mat = mat.clone();

		addSignature(show_mat);

		Util.showImage("Source Image", show_mat);

		// Process canny edge dectection
		Mat canny_mat = Mat.zeros(mat.size(), mat.type());

		Imgproc.Canny(mat, canny_mat, CANNY_THRESHOLD * 1, CANNY_THRESHOLD * 3);

		Mat show_canny = canny_mat.clone();

		addSignature(show_canny);

		Util.showImage("Canny Edge Detection", show_canny);

		// Process hough transform
		Mat hough_mat = Mat.zeros(mat.size(), mat.type());

		Mat hough_line = new Mat();

//		Imgproc.HoughLines(canny_mat, hough_line, 1, Math.PI / 180, 0);
		Imgproc.HoughLinesP(canny_mat, hough_line, 1, Math.PI / 180, 50, 50, 10);
//		Imgproc.HoughLines(canny_mat, hough_line, RHO, THETA, THRESHOLD);

		for (int x = 0; x < hough_line.rows(); x++) {
			
			double[] l = hough_line.get(x, 0);
			Imgproc.line(hough_mat, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 255, 255), 3, Imgproc.LINE_AA, 0);
		}

		Mat show_hough = hough_mat.clone();

		addSignature(show_hough);

		Util.showImage("Hough Transform", show_hough);
	}

	public static void addSignature(Mat source_mat) {

		try {

			int source_start_x = source_mat.cols() - 1 - 64;
			int source_start_y = source_mat.rows() - 1 - 32;

			Mat signature_mat = Util.image2cvMat(ImageIO.read(new File(ROOT + "./opencv/signature.jpg")));

			Imgproc.cvtColor(signature_mat, signature_mat, Imgproc.COLOR_BGR2GRAY);

			for (int i = 0; i < 64; i++) {
				for (int j = 0; j < 32; j++) {

					byte[] data = new byte[signature_mat.channels()];

					signature_mat.get(j, i, data);

					if ((data[0] & 0xFF) <= 8) {

						continue;
					}

					byte[] source_data = new byte[source_mat.channels()];

					source_mat.get(source_start_y + j, source_start_x + i, source_data);

					for (int k = 0; k < source_data.length; k++) {

						source_data[k] = data[0];
					}

					source_mat.put(source_start_y + j, source_start_x + i, source_data);
				}
			}
		}
		catch (Exception e) {

			return;
		}
	}

	public static byte getPixelDepth(Mat source_mat, int x, int y, byte default_depth) {

		int cols = source_mat.cols();
		int rows = source_mat.rows();

		if (x < rows || rows <= x || y < cols || cols <= y) {

			return default_depth;
		}

		/*
		 * if (source_mat.type() != CV_TYPE) {
		 * 
		 * return default_depth; }
		 */

		byte[] data = new byte[1];

		source_mat.get(x, y, data);

		return data[0];
	}
}
