import argparse
import cv2
import onnxruntime
import numpy as np


class C2PNet:
    def __init__(self, modelpath):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input_name = self.onnx_session.get_inputs()[0].name
        _, _, self.input_height, self.input_width = self.onnx_session.get_inputs()[0].shape

    def detect(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(self.input_height ,int) and isinstance(self.input_width, int):
            input_image = cv2.resize(input_image, (self.input_width, self.input_height)) ###固定输入分辨率, HXW.onnx文件是动态输入分辨率的
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)

        result = self.onnx_session.run(None, {self.input_name: input_image}) ###opencv-dnn推理时,结果图全黑
        
        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        output_image = np.squeeze(result[0])
        output_image = output_image.transpose(1, 2, 0)
        output_image = output_image * 255
        output_image = np.clip(output_image, 0, 255)
        output_image = output_image.astype(np.uint8)
        output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))
        return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str,
                        default='testimgs/outdoor/0143_1_0.2.jpg', help="image path")
    parser.add_argument('--modelpath', type=str,
                        default='weights/c2pnet_outdoor_360x640.onnx', help="image path")
    args = parser.parse_args()

    mynet = C2PNet(args.modelpath)
    srcimg = cv2.imread(args.imgpath)

    dstimg = mynet.detect(srcimg)

    if srcimg.shape[0] > srcimg.shape[1]:
        boundimg = np.zeros((10, srcimg.shape[1], 3), dtype=srcimg.dtype)+255  ###中间分开原图和结果
        combined_img = np.vstack([srcimg, boundimg, dstimg])
    else:
        boundimg = np.zeros((srcimg.shape[0], 10, 3), dtype=srcimg.dtype)+255
        combined_img = np.hstack([srcimg, boundimg, dstimg])
    cv2.imwrite('result.jpg', combined_img)
    winName = 'Deep learning Image Dehaze use onnxruntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, combined_img)  ###原图和结果图也可以分开窗口显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
