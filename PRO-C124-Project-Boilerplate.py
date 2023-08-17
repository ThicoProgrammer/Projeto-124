# Para capturar os quadros
import cv2

# Para processar o array de imagens
import numpy as np

# importe os módulos tensorflow e carregue o modelo
import tensorflow as tf

modelo = tf.keras.models.load_model('keras_model.h5')

# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

# Loop infinito
while True:

    # Lendo / requisitando um quadro da câmera 
    status, frame = camera.read()

    # Se tivemos sucesso ao ler o quadro
    if status:

        # Inverta o quadro
        frame = cv2.flip(frame, 1)

        # Redimensione o quadro
        frame_redimensionado = cv2.resize(frame, (224, 224))

        # Expanda a dimensão do array junto com o eixo 0
        frame_expandido = np.expand_dims(frame_redimensionado, axis=0)

        # Normalize para facilitar o processamento
        frame_normalizado = frame_expandido / 255.0

        # Obtenha previsões do modelo
        previsoes = modelo.predict(frame_normalizado)
        classe_predita = np.argmax(previsoes)

        # Defina a legenda com base na classe predita
        if classe_predita == 0:
            legenda = "Pedra"
        elif classe_predita == 1:
            legenda = "Papel"
        else:
            legenda = "Tesoura"

        # Exibindo os quadros capturados com a legenda
        cv2.putText(frame, legenda, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('feed', frame)

        # Aguardando 1ms
        code = cv2.waitKey(1)

        # Se a barra de espaço foi pressionada, interrompa o loop
        if code == 32:
            break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()