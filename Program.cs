using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace Lines
{
    class Program
    {
        static void Main(string[] args)
        {
            // Загрузка классификаторов «HaarCascadeClassifier» для распознавания лиц
            var faceCascadeClassifier = new CascadeClassifier("haarcascade_frontalface_default.xml");
            var eyeCascadeClassifier = new CascadeClassifier("haarcascade_eye.xml");

            // Загрузка изображения, на котором необходимо распознать лицо
            var image = new Image<Bgr, byte>("img.pgm");

            // Преобразование изображения в полутоновое
            var grayImage = image.Convert<Gray, byte>();

            // Распознавание лиц на полутоновом изображении
            var faces = faceCascadeClassifier.DetectMultiScale(grayImage, 1.1, 5);
            var eyes = eyeCascadeClassifier.DetectMultiScale(grayImage, 1.1, 5);

            int center_x;
            int center_y;
            int eye_x;
            int eye_y;

            // Прорисовка центральной линии симметрии
            foreach (var face in faces)
            {
                center_x = Convert.ToInt32(face.X + face.Width * 0.5);
                center_y = Convert.ToInt32(face.Y + face.Height * 0.5);
                CvInvoke.Line(image, new Point(center_x, center_y + 1000), new Point(center_x, center_y - 1000), new MCvScalar(0, 0, 255), 1, LineType.EightConnected);

                // Прорисовка локальных линий симметрии
                foreach (var eye in eyes)
                {
                    eye_x = Convert.ToInt32(eye.X + eye.Width * 0.5);
                    eye_y = Convert.ToInt32(eye.Y + eye.Height * 0.5);
                    CvInvoke.Line(image, new Point(eye_x, eye_y + 100), new Point(eye_x, eye_y - 100), new MCvScalar(0, 255, 0), 1, LineType.EightConnected);
                    double distance = Math.Sqrt(Math.Pow(center_x - eye_x, 2) + Math.Pow(center_y - eye_y, 2));
                    Console.WriteLine(distance);
                }
            }
            CvInvoke.Resize(image, image, new Size(Convert.ToInt32(image.Width * 4), Convert.ToInt32(image.Height * 4)));
            CvInvoke.Imshow(" Lines of symmetry", image);
            CvInvoke.WaitKey();
        }
    }
}
