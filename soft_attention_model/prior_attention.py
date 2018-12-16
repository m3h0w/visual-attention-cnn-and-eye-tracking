import numpy as np
import scipy

def get_bins(T):
    return list(list(range(1+int(np.sqrt(T))))/np.sqrt(T))

def fixation_to_bin(fixation_coordinate, bins):
    return sorted(bins + [fixation_coordinate]).index(fixation_coordinate) - 1

def _convert_fixation_to_indices(fixation, bins):
    # assuming x,y
    col = fixation_to_bin(fixation[0], bins)
    row = fixation_to_bin(fixation[1], bins)
    return row, col

def _convert_indices_to_bin(row, col, T):
    return row*int(np.sqrt(T)) + col

def convert_fixation_to_bin(fixation, T):
    row, col = _convert_fixation_to_indices(fixation, get_bins(T))
    bin_n = _convert_indices_to_bin(row, col, T)
    return bin_n

def build_prior_attention(fixation, T):
    columns = int(np.sqrt(T))
    row, col = _convert_fixation_to_indices(fixation, get_bins(T))
    bin_n = convert_fixation_to_bin(fixation, T)
    attention = np.zeros(shape=(columns,columns))
    attention[row, col] = 1
    return attention

def build_prior_attentions(fixations, T, lstm_time, gauss_sigma=None):
    prior_attentions = []
    for fixation_series in fixations:
        prior_attention_series = []
        for t in range(lstm_time):
            if t >= fixation_series.shape[0]:
                fixation = np.mean(fixation_series, axis=0)
            else:
                fixation = fixation_series[t]
            attention_vector = build_prior_attention(fixation, T)
            if gauss_sigma:
                attention_vector = scipy.ndimage.filters.gaussian_filter(attention_vector, sigma=gauss_sigma)
            prior_attention_series.append(attention_vector.reshape(T,))
        prior_attentions.append(prior_attention_series)
    return np.array(prior_attentions)
