import cv2
import numpy as np
import sqlite3

# Rutas de los archivos necesarios
base_dir = "D:/ProyectoCamaraPersonas/"
weights_path = base_dir + "yolov3-tiny.weights"
config_path = base_dir + "yolov3-tiny.cfg"
names_path = base_dir + "coco.names"

# Inicializar el modelo de detección YOLOv3-Tiny
net = cv2.dnn.readNet(weights_path, config_path)
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Inicializar contador
person_count = 0

# Conectar a la base de datos
conn = sqlite3.connect("person_counter.db")
cursor = conn.cursor()

# Crear tabla para almacenar información (si no existe)
cursor.execute('''CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    count INTEGER
                )''')
conn.commit()

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Si tienes varias cámaras, puedes cambiar el índice (por ejemplo, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Realizar detección de objetos (incluyendo personas) con YOLOv3-Tiny
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar las detecciones
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 representa "persona"
                # Procesar las coordenadas de la caja delimitadora (bounding box)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas de la esquina superior izquierda de la caja delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Dibujar la caja delimitadora y mostrar el texto
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Persona", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Incrementar el contador de personas
                person_count += 1

    # Almacenar la información en la base de datos
    cursor.execute("INSERT INTO people (count) VALUES (?)", (person_count,))
    conn.commit()

    # Mostrar el contador en el marco de video
    cv2.putText(frame, f"Contador: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el video en tiempo real
    cv2.imshow("Detección de Personas", frame)

    # Detener la detección si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
conn.close()
