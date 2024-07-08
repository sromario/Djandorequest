import cv2
from ultralytics import YOLO

# def para fazer a inferência yolo
def predict_image(model, image_path, conf=0.7, iou=0.5, imgsz=640):
    # Ler a imagem 
    image = cv2.imread(image_path)
    
    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return False, None  # Retorna False se houver erro ao carregar a imagem

    # Realizar a inferência na imagem
    results_img = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=False
    )
    
    # Verificar se houve detecção
    detections = results_img[0].boxes if results_img else None
    if detections:
        for box in detections:
            if box.conf > conf:
                return True, results_img  
    
    return False, results_img 

# Função para desenhar caixas delimitadoras na imagem e salvar a imagem resultante
def draw_boxes(image_path, results_img, model, output_path):
    # Ler a imagem 
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return False

    # Desenhar caixas delimitadoras na imagem
    detections = results_img[0].boxes if results_img else None
    if detections:
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[int(box.cls)]}: {float(box.conf):.2f}'  # Converte box.conf para float
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
            # Salvar a imagem com as detecções
            cv2.imwrite(output_path, image)
    return True
