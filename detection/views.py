from django.shortcuts import render
from django.http import JsonResponse
from detection.yolo_detector.utils import predict_image, draw_boxes
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import os 

model = YOLO('detection/yolo_detector/weights/best.pt')

#receber post request
@csrf_exempt
def detect_objects_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Salvar a imagem recebida // adaptar ao banco aws
        temp_path = 'temp.jpg'
        with open(temp_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        # inferência yolo
        detected, results_img = predict_image(model, temp_path, conf=0.7, iou=0.5)
        
        # Desenhar caixas caso tenha deteção e remover foto original, salvando apenas nova com deteção
        if detected:
            output_path = 'detected_temp.jpg'
            draw_boxes(temp_path, results_img, model, output_path)
            os.remove(temp_path)
            print(f"Imagem original excluída: {temp_path}")
            
        
       
        return JsonResponse({'detected': detected})

    
    return JsonResponse({'error': 'Envie uma imagem válida'}, status=400)
