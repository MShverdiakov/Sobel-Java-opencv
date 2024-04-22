import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class SobelRun {

   public void run(String[] args) {

      Mat src, src_gray = new Mat();
      Mat grad = new Mat();
      int scale = 1;
      int delta = 0;
      int ddepth = CvType.CV_16S;

      src = Imgcodecs.imread("./flower.png");

      if( src.empty() ) {
         System.out.println(args[0]);
         System.exit(-1);
      }

      Imgproc.GaussianBlur( src, src, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT );

      Imgproc.cvtColor( src, src_gray, Imgproc.COLOR_RGB2GRAY );

      Mat grad_x = new Mat(), grad_y = new Mat();
      Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

      Imgproc.Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, Core.BORDER_DEFAULT );

      Imgproc.Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, Core.BORDER_DEFAULT );

      Core.convertScaleAbs( grad_x, abs_grad_x );
      Core.convertScaleAbs( grad_y, abs_grad_y );

      Core.addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

      HighGui.imshow("result", grad );
      HighGui.waitKey(0);

      System.exit(0);
   }
}

public class Sobel {
   public static void main(String[] args) {
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
      new SobelRun().run(args);
   }
}