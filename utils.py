import numpy as np
import os


def normalization(ordered_lines):
    """
    Segments are normalized so that the draw occupies all the canvas.
    Draw touches both the horizontal or vertical borders.
    Draw is centered.
    Segments are normalized in [0,1]
    :param ordered_lines: segment matrix (N+2)x4
    :return: normalized segment matrix (N+2)x4
    """

    maxs = np.max(ordered_lines, axis=0)
    mins = np.min(ordered_lines, axis=0)
    xmax = max(maxs[0], maxs[2])
    ymax = max(maxs[1], maxs[3])
    xmin = min(mins[0], mins[2])
    ymin = min(mins[1], mins[3])

    # data scaling
    if (xmax - xmin) > (ymax - ymin):
        select_axis_x = True
    else:
        select_axis_x = False

    if select_axis_x:
        z_matrix = (ordered_lines - xmin)
        z_matrix = z_matrix / (xmax - xmin)
    else:
        z_matrix = (ordered_lines - ymin)
        z_matrix = z_matrix / (ymax - ymin)

    # data centering
    maxs = np.max(z_matrix, axis=0)
    mins = np.min(z_matrix, axis=0)
    xmax = max(maxs[0], maxs[2])
    ymax = max(maxs[1], maxs[3])
    xmin = min(mins[0], mins[2])
    ymin = min(mins[1], mins[3])

    if select_axis_x:
        t = (1.0 - (ymax - ymin)) / 2.0
        z_matrix[:, 1] = z_matrix[:, 1] + t - ymin
        z_matrix[:, 3] = z_matrix[:, 3] + t - ymin
    else:
        t = (1.0 - (xmax - xmin)) / 2.0
        z_matrix[:, 0] = z_matrix[:, 0] + t - xmin
        z_matrix[:, 2] = z_matrix[:, 2] + t - xmin

    return z_matrix


def create_minibatches(x, batch_size):
    """
    Create a list of batch indeces
    :param x: numpy array data
    :param batch_size: batch size
    :return: list of arrays of batch indices
    """
    n = x.shape[0]  # number of examples
    batch_idxs = []
    idxs = np.arange(0, n)
    np.random.shuffle(idxs)
    for i in range(0, n, batch_size):
        if batch_size + i < n:
            batch_idx = idxs[i:batch_size+i]
        else:
            batch_idx =idxs[i:]
        batch_idxs.append(batch_idx)
    return batch_idxs


def find_last_checkpoint(checkpoint_dir, model_name):
    list_numbers = []
    for file in os.listdir(checkpoint_dir):
        if file != 'checkpoint':
            end = file.split('-')[-1]
            if end != '00001':
                if end.split('.')[1] == 'meta':
                    number = end.split('.')[0]
                    list_numbers.append(number)

    max_number = max(list_numbers)
    last_checkpoint = model_name + '-' + str(max_number)
    return last_checkpoint


def random_augmentation(z_matrix, range):
    """
    Perturb each coordinate of a random value in (-range, range)
    :param z_matrix: normalized in [0,1] segments coordinates [x0,y0,x1,y1]
    :param range: float
    :return: perturbed segments
    """
    points = np.reshape(z_matrix, (z_matrix.shape[0] * 2, 2))
    points_unique, points_ids, points_inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    perturbations = np.random.uniform(low=-range, high=range, size=(points.shape[0], 2))
    z_matrix += np.reshape(perturbations[points_ids[points_inverse], :], z_matrix.shape)
    #z_matrix += np.reshape(perturbations[points_inverse, :], z_matrix.shape)
    return z_matrix


def get_perturbed_matrix(matrix, N, range):
    """
    Perturb each coordinate of a segments matrix of a random value in (-range, range)
    :param matrix: segments matrix (N+2,4) normalized in [0,1]
    :param N: number maximum of segments
    :param range: float
    :return: perturbed matrix (N+2,4) normalized in [0,1]
    """
    sum_coord = np.sum(matrix, axis=1)
    n = sum_coord[sum_coord != 0].shape[0] - 1  # number of segments
    segments = matrix[1:n + 1, :]  # segments coordinates
    pert_segments = random_augmentation(segments, range)
    norm_segments = normalization(pert_segments)
    norm_matrix = np.zeros((N + 2, 4))
    norm_matrix[1:n + 1, :] = norm_segments
    norm_matrix[n + 1, :] = 1  # end
    return norm_matrix

#index from 0 to 49
def scale_color(i):
    if i < 10:
        r = int(i*2.5)
        g = 0
        b = 0
    if i >= 10 and i < 20:
        r = 255
        g = 0
        b = int((i-10)*2.5)
    if i >= 20 and i < 30:
        r = int((30-i)*2.5)
        g = 0
        b = 255
    if i >= 30 and i < 40:
        r = 0
        g = int((i-30)*2.5)
        b = 255
    if i >= 40:
        r = 0
        g = 255
        b = int((50-i)*2.5)
    return r,g,b



