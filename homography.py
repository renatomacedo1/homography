import os
import cv2
import numpy as np

def are_same_object(img1, img2, kp1, kp2, good_matches, min_matches=4):
    """Verifica se duas imagens contêm o mesmo objeto através da homografia."""
    if len(good_matches) < min_matches:
        return False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M is not None

def get_image_pairs(directory):
    """Pega todas as imagens no diretório especificado e as retorna em pares."""
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_pairs = []

    for i in range(len(image_files)):
        for j in range(i+1, len(image_files)):
            image_pairs.append((os.path.join(directory, image_files[i]), os.path.join(directory, image_files[j])))

    # Usando SIFT como detector e descriptor
    sift = cv2.SIFT_create()

    for img_path1, img_path2 in image_pairs:
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            print(f"Não foi possível abrir {img_path1} ou {img_path2}. Pulando este par.")
            continue


        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Não foram encontrados suficientes keypoints em {img_path1} ou {img_path2}. Pulando este par.")
            continue

        # Usando FLANN como matcher, que é mais adequado para SIFT/SURF
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)

        # Filtrando as correspondências usando a razão de Lowe
        good_matches = []
        for m, n in matches:
            if m.distance < 0.78 * n.distance:
                good_matches.append(m)

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
        cv2.imshow(f'Matches between {img_path1} and {img_path2}', match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        if are_same_object(img1, img2, kp1, kp2, good_matches):
            print(f'{img_path1} e {img_path2} contêm o mesmo objeto!')
        else:
            print(f'{img_path1} e {img_path2} NÃO contêm o mesmo objeto!')

# Testando o programa
directory = 'imgs'
get_image_pairs(directory)
