#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// Include the header for your monogenicProcessor class
// Make sure the include path is correct relative to your project structure
#include "monogenicProcessor.h"
#include <iostream>
#include <string> // Required for std::string

// This example demonstrates the basic usage of the monogenicProcessor
// class. It expects as the only command line argument, the name of a single
// image file (e.g., .png, .jpg, .bmp). It calculates the 2D monogenic signal
// representation of the image and displays the even part and the two odd parts
// on the screen. It also calculates the feature symmetry and asymmetry measures,
// and displays these too.

// The monogenic signal is a representation of single channel (greyscale) images,
// so if the supplied image is colour, it will be converted to greyscale before
// processing.

// Namespaces
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	// Check that the user has supplied a single argument, which is the image
	// file to use for the demonstration
	if( argc != 2)
	{
		cout << " Usage: " << argv[0] << " <imagefilename>" << endl;
		return -1;
	}
	const string imgname = argv[1];

	// Load the image file
	// cv::IMREAD_GRAYSCALE ensures the image is loaded as grayscale
	Mat input_image = imread(imgname, cv::IMREAD_GRAYSCALE);

	if ( input_image.empty() ) // Check for invalid input
	{
		cout << "Could not open or find the image: " << imgname << endl;
		return -1;
	}

	// Get dimensions of the image
	const int xsize = input_image.cols;
	const int ysize = input_image.rows;

	// Create display windows for the monogenic signal components and derived measures
	namedWindow( "Even", WINDOW_AUTOSIZE );
	namedWindow( "Odd Y", WINDOW_AUTOSIZE );
	namedWindow( "Odd X", WINDOW_AUTOSIZE );
	namedWindow( "Feature Symmetry", WINDOW_AUTOSIZE );
	namedWindow( "Feature Asymmetry", WINDOW_AUTOSIZE );

	// The first step to using the monogenicProcessor is to initialise
	// a monogenic processor object. At a minimum we must provide the dimensions
	// of the input image and a centre-wavelength to use for the log Gabor filter.
	// The shorter the wavelength, the more fine detail is preserved.
	// We'll choose 50 pixels, for no particular reason.
	// Once created this object must only be used with images of the matching
	// size.
	monogenic::monogenicProcessor mgFilts(ysize, xsize, 50);

	// Declare matrices to hold the results
	Mat even, oddy, oddx, fs, fa;
	Mat disp1, disp2, disp3, disp4, disp5; // Matrices for normalized display images

	// This line performs the calculation on the input_image to find the monogenic
	// signal representation. This must be performed before trying to access
	// the components of the monogenic signal or any of the derived measures
	// (such as feature symmetry).
	mgFilts.findMonogenicSignal(input_image);

	// Now we can access the odd and even components of the monogenic signal
	// representation of the image.
	mgFilts.getEvenFilt(even);
	mgFilts.getOddFiltCartesian(oddy,oddx);

	// We can also access some quantities derived from the monogenic signal,
	// such as feature symmetry (fs) and feature asymmetry (fa).
	mgFilts.getFeatureSymmetry(fs);
	mgFilts.getFeatureAsymmetry(fa);

	// Display even and odd components
	// Normalize the output for better visualization (values might be outside 0-255)
	normalize(even, disp1, 0, 1, cv::NORM_MINMAX);
	imshow("Even", disp1);

	normalize(oddy, disp2, 0, 1, cv::NORM_MINMAX);
	imshow("Odd Y", disp2);

	normalize(oddx, disp3, 0, 1, cv::NORM_MINMAX);
	imshow("Odd X", disp3);

	// Display feature symmetry and asymmetry
	// These measures are typically in the range [0, 1] or [-1, 1], normalize for display
	normalize(fs, disp4, 0, 1, cv::NORM_MINMAX); // Assuming symmetry is [0, 1] or similar
	imshow("Feature Symmetry", disp4);

	normalize(fa, disp5, 0, 1, cv::NORM_MINMAX); // Assuming asymmetry is [0, 1] or similar
	imshow("Feature Asymmetry", disp5);

	// Wait indefinitely until a key is pressed to keep the windows open
	cout << "Press any key to exit." << endl;
	waitKey(0);

	return 0;
}
