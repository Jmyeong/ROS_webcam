import cv2
import numpy as np
import os

# 모델 파일과 클래스 정보 파일 경로
model_file = './enet-cityscapes/enet-model.net'
class_info_file = './enet-cityscapes/enet-classes.txt'
# 클래스 정보 파일에서 클래스 이름 읽어오기
with open(class_info_file, 'r') as f:
    class_names = f.read().strip().split('\n')

# ENet 모델 로드
enet_neural_network = cv2.dnn.readNet(model_file)

# 노트북 카메라 객체 생성
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # 프레임 크기를 ENet 입력 크기에 맞게 리사이즈
    input_frame = cv2.resize(frame, (1024, 512))

    # Semantic Segmentation 결과 적용
    input_img_blob = cv2.dnn.blobFromImage(input_frame, 1.0 / 255, (1024, 512), 0, swapRB=True, crop=False)
    enet_neural_network.setInput(input_img_blob)
    enet_neural_network_output = enet_neural_network.forward()

    road_class_index = class_names.index('Road')  # 'Road' 클래스의 인덱스 가져오기
    road_class_map = np.argmax(enet_neural_network_output[0], axis=0) == road_class_index

    road_mask = np.zeros_like(input_frame)  # road_mask를 input_frame과 같은 크기로 생성
    road_mask[road_class_map] = input_frame[road_class_map]

    class_map = np.argmax(enet_neural_network_output[0], axis=0)

    if os.path.isfile('./enet-cityscapes/enet-colors.txt'):
        IMG_COLOR_LIST = (open('./enet-cityscapes/enet-colors.txt').read().strip().split("\n"))
        IMG_COLOR_LIST = [np.array(color.split(",")).astype("int") for color in IMG_COLOR_LIST]
        IMG_COLOR_LIST = np.array(IMG_COLOR_LIST, dtype="uint8")
    else:
        np.random.seed(1)
        IMG_COLOR_LIST = np.random.randint(0, 255, size=(len(class_names) - 1, 3), dtype="uint8")
        IMG_COLOR_LIST = np.vstack([[0, 0, 0], IMG_COLOR_LIST]).astype("uint8")

    class_map_mask = IMG_COLOR_LIST[class_map]

    class_map_mask = cv2.resize(class_map_mask, (input_frame.shape[1], input_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    enet_neural_network_output = ((0.60 * class_map_mask) + (0.40 * input_frame)).astype("uint8")

    class_legend = np.zeros(((len(class_names) * 25) + 25, 300, 3), dtype="uint8")

    for (i, (cl_name, cl_color)) in enumerate(zip(class_names, IMG_COLOR_LIST)):
            color_information = [int(color) for color in cl_color]
            cv2.putText(class_legend, cl_name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(class_legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color_information), -1)

    combined_images = np.concatenate((input_frame, enet_neural_network_output), axis=1)
    # 결과 표시
    cv2.imshow("Semantic Segmentation", road_mask)
    cv2.imshow('Results', combined_images)
    cv2.imshow("Class Legend", class_legend)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv2.destroyAllWindows()
