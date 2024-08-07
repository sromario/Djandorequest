from django.shortcuts import render
from django.http import JsonResponse
from detection.yolo_detector.utils import predict_image, preprocess_image, rotacionar_imagem
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import cv2
import os

# Carregar o modelo YOLO
model = YOLO('detection/yolo_detector/weights/best.pt')

@csrf_exempt
def detect_objects_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Salvar a imagem recebida
        temp_path = 'temp.jpg'
        with open(temp_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        try:
            # Rotacionar a imagem antes do pré-processamento
            rotacionar_imagem(temp_path)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)

        # Ler a imagem rotacionada e pré-processar
        image = cv2.imread(temp_path)
        if image is None:
            return JsonResponse({'error': 'Erro ao carregar a imagem rotacionada'}, status=400)
        
        prepro_image = preprocess_image(image)

        # Realizar a inferência
        isPromoBottle, results_img = predict_image(model, prepro_image, conf=0.75, iou=0.5)
        
        # Retornar o valor booleano como JSON
        return JsonResponse({'isPromoBottle': isPromoBottle})
        
    return JsonResponse({'error': 'Envie uma imagem válida'}, status=400)
