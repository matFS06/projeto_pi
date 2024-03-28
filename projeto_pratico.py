import cv2
import numpy as np

def ler_imagem_pbm(nome_arquivo):
    with open(nome_arquivo, 'r') as f:
        linhas = f.readlines()
    imagem_binaria = []
    for linha in linhas[3:]:
        imagem_binaria.append([int(x) for x in linha.strip()])
    return np.array(imagem_binaria)

def encontrar_linhas(imagem):
    # Erosão para eliminar ruídos sal e pimenta e manter apenas linhas
    kernel = np.ones((5, 5), np.uint8)
    imagem_erodida = cv2.erode(imagem, kernel, iterations=1)
    
    # Contagem de linhas usando a transformada de Hough
    linhas = cv2.HoughLinesP(imagem_erodida, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    return linhas

def encontrar_palavras(imagem):
    # Dilatação para unir caracteres e formar palavras
    kernel = np.ones((5, 5), np.uint8)
    imagem_dilatada = cv2.dilate(imagem, kernel, iterations=1)
    
    # Identificar contornos
    contornos, _ = cv2.findContours(imagem_dilatada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Aproximar retângulos para os contornos encontrados
    retangulos = [cv2.boundingRect(contorno) for contorno in contornos]
    
    return retangulos

def desenhar_retangulos(imagem, retangulos):
    for x, y, w, h in retangulos:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 255, 255), 1)

def contar_linhas_e_palavras(linhas, palavras):
    return len(linhas), len(palavras)

def main():
    # Ler a imagem PBM
    imagem = ler_imagem_pbm('imagem.pbm')
    
    # Encontrar linhas e palavras
    linhas = encontrar_linhas(imagem)
    palavras = encontrar_palavras(imagem)
    
    # Contar linhas e palavras
    num_linhas, num_palavras = contar_linhas_e_palavras(linhas, palavras)
    
    # Desenhar retângulos ao redor das palavras na imagem original
    imagem_original = cv2.cvtColor(imagem.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    desenhar_retangulos(imagem_original, palavras)
    
    # Mostrar resultado
    cv2.imshow('Imagem com retângulos', imagem_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Número de linhas:", num_linhas)
    print("Número de palavras:", num_palavras)

if __name__ == "__main__":
    main()
