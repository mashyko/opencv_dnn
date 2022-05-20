import numpy as np
import cv2 as cv

# ファイルからクラスの名前のリストを読み込む関数
def read_classes(file):
    classes = None
    with open(file, mode='r', encoding="utf-8") as f:
        classes = f.read().splitlines()
    return classes

# クラスの数だけカラーテーブルを生成する関数
def get_colors(num):
    colors = []
    np.random.seed(0)
    for i in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors

def main():
    # キャプチャを開く
    capture = cv.VideoCapture('dog.jpg')
    #capture = cv.VideoCapture(0) # カメラ
    if not capture.isOpened():
        exit()

    # モデルを読み込む
    weights = "yolov4-mish.weights" # 640x640
    config = "yolov4-mish.cfg"
    model = cv.dnn_DetectionModel(weights, config)

    # モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # モデルの入力パラメーターを設定する
    scale = 1.0 / 255.0    # スケールファクター
    size = (640, 640)      # 入力サイズ（CSP、MISH）
    mean = (0.0, 0.0, 0.0) # 差し引かれる平均値
    swap = True            # チャンネルの順番（True: RGB、False: BGR）
    crop = False           # クロップ
    model.setInputParams(scale, size, mean, swap, crop)

    # NMS（Non-Maximum Suppression）をクラスごとに処理する
    model.setNmsAcrossClasses(False) # （True: 全体、False: クラスごと）

    # クラスリストとカラーテーブルを取得する
    names = "coco.names"
    classes = read_classes(names)
    colors = get_colors(len(classes))

    while True:
        # フレームをキャプチャして画像を読み込む
        result, image = capture.read()
        if result is False:
            cv.waitKey(0)
            break

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

        # オブジェクトを検出する
        confidence_threshold = 0.5
        nms_threshold = 0.4
        class_ids, confidences, boxes = model.detect(image, confidence_threshold, nms_threshold)

        # 2次元配列(n, 1)から1次元配列(n, )に変換
        class_ids = np.array(class_ids).flatten()
        confidences = np.array(confidences).flatten()

        # 検出されたオブジェクトを描画する
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            class_name = classes[class_id]
            color = colors[class_id]
            thickness = 2
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)

            result = "{0} ({1:.3f})".format(class_name, confidence)
            point = (box[0], box[1] - 5)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            cv.putText(image, result, point, font, scale, color, thickness, cv.LINE_AA)

        # 画像を表示する
        cv.imshow("object detection", image)
        key = cv.waitKey(10)
        if key == ord('q'):
            break    
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()