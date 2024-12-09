# Weight Voronoi Image Stippling

## Project Title
**Weight Voronoi Image Stippling**

## Team Members
- Mary Akhila Reddy Kampally
- Tharun Macha

## Software Used
- Python 3.x
- Streamlit
- OpenCV
- NumPy
- Pillow
- Scikit-learn
- SciPy
- Matplotlib

## Installation Guide

### Requirements
To run the project, you need to install the required dependencies. These can be installed using the `requirements.txt` file.

1. Clone the repository:
    ```bash
    git clone https://github.com/mkampally/Weight-Voronoi-Image-Stippling
    cd Weight Voronoi Image Stippling
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Run the Program
To run the application:

1. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

2. This will start a local server, and you can access the application in your web browser at:
    ```
    http://localhost:8501
    ```

### Expected Output
Once the application is running:

1. Upload an image (JPG, JPEG, or PNG format).
2. The program will display the uploaded image in grayscale.
3. You can adjust various stippling parameters using the sidebar, including the number of points, the stippling radius, and more.
4. Once you click "Generate Stippled Image", the application will process the image and show the stippled output, as well as an option to view the Voronoi diagram.

### Input Image Format and Size
- The program accepts images in the following formats: `.jpg`, `.jpeg`, `.png`.
- It works with images of any size, though larger images may take longer to process.

### Operating System
The program should work on both **Windows** and **Linux** operating systems.

## Code References and Resources

The algorithm used in this project for Voronoi-based stippling was inspired by the following reference:

- Secord, A. (2002). *Stippling Using Voronoi Diagrams*. [Link to Paper](https://www.cs.ubc.ca/labs/imager/tr/2002/secord2002b/secord.2002b.pdf)

Additional resources that helped with the implementation:
- YouTube video tutorial explaining Voronoi stippling algorithm in JavaScript: [Watch Video](https://www.youtube.com/watch?v=Bxdt6T_1qgc)

## Comments and Code Explanation

- The main logic of the program consists of generating a Voronoi diagram from KMeans clustering centroids based on the grayscale image.
- The stippled image is generated by placing stipple points (black dots) at the centroids of the Voronoi regions.
- The `generate_stippled_image()` function takes care of creating a canvas and adding stipples to it, while the `calculate_centroids()` function computes the centroid of each Voronoi region.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to all the developers and researchers whose work inspired this project.
- Special thanks to [Streamlit](https://streamlit.io/), [OpenCV](https://opencv.org/), and [Scikit-learn](https://scikit-learn.org/) for their amazing libraries, which made the implementation much easier.
