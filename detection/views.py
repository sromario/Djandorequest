from django.shortcuts import render
from django.http import JsonResponse
from detection.yolo_detector.utils import predict_image, preprocess_image
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
        
        # Ler a imagem e pré-processar
        image = cv2.imread(temp_path)
        if image is None:
            return JsonResponse({'error': 'Erro ao carregar a imagem'}, status=400)
        
        prepro_image = preprocess_image(image)
        
        # Realizar a inferência
        isPromoBottle, results_img = predict_image(model, prepro_image, conf=0.75, iou=0.5)
        
        # Verificar se houve detecção
        if isPromoBottle:
            return JsonResponse({'detected': True})
        else:
            return JsonResponse({'detected': False})
        
    return JsonResponse({'error': 'Envie uma imagem válida'}, status=400)