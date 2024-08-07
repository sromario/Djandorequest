import cv2
from ultralytics import YOLO

def preprocess_image(image, target_size=(640, 640)):
    """
    Pré-processa a imagem para o modelo YOLO.

    Returns:
    - numpy.ndarray: Imagem pré-processada.
    """
    # Redimensionar a imagem
    image_resized = cv2.resize(image, target_size)
    
    # Converter para RGB (OpenCV usa BGR por padrão)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalizar a imagem
    image_normalized = image_rgb / 255.0

    # Converter para formato esperado pelo YOLO
    image_preprocessed = (image_normalized * 255).astype('uint8')
    
    return image_preprocessed

def predict_image(model, image, conf=0.7, iou=0.5, imgsz=640):
    """
    Realiza a inferência usando o modelo YOLO.

    Returns:
    - bool: Verdadeiro se alguma detecção com confiança > conf for encontrada.
    - results_img (Results): Resultados da inferência.
    """
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
            if box.conf[0] > conf:
                return True, results_img
    
    return False, results_img

def rotacionar_imagem(caminho_imagem):
    """
    Rotaciona a imagem em 90 graus no sentido anti-horário e salva a imagem rotacionada.

    """
    # Carrega a imagem
    imagem = cv2.imread(caminho_imagem)

    # Verifica se a imagem foi carregada corretamente
    if imagem is None:
        raise ValueError("Erro ao carregar a imagem para rotação")

    # Rotaciona a imagem 90 graus no sentido anti-horário
    imagem_rotacionada = cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Salva a imagem rotacionada
    cv2.imwrite(caminho_imagem, imagem_rotacionada)
