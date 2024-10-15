import numpy as np
from scipy import interpolate
import copy
from PyEMD import EMD
from scipy.signal import square
import math

def smooth_signal(signal, window_size=5):
    """
    Smooth the signal using a simple moving average.

    Parameters:
    signal (numpy.array): Input signal to smooth.
    window_size (int): The number of points to use in the moving average window.

    Returns:
    numpy.array: Smoothed signal.
    """
    if window_size < 2:
        return signal

    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='valid')

    # Handle edge cases by padding the start and end of the signal
    padding_before = np.full((window_size // 2,), signal[0])
    padding_after = np.full((window_size - 1 - window_size // 2,), signal[-1])

    smoothed_signal = np.concatenate((padding_before, smoothed_signal, padding_after))

    return smoothed_signal


def extend_array_to_target_length(arr, target_length):
    """
    Extend an array to a target length by repeating its elements.

    Parameters:
    arr (numpy.array): The original array to extend.
    target_length (int): The target length for the extended array.

    Returns:
    numpy.array: The extended array.
    """
    original_length = len(arr)
    multiple = math.ceil(target_length / original_length)
    result = np.zeros(original_length * multiple, dtype=arr.dtype)
    result[::multiple] = arr
    return result



class PassArgs:
    def __init__(self):
        self.RescalFactor = 1


def IFFTB(mean_magnitudes, std_magnitudes, mean_phases, std_phases, state='normal', args=None, size=100, discountFactor=None):
    """
    Generate a new signal based on the provided magnitudes and phases using FFT.

    Parameters:
    mean_magnitudes (numpy.array): Mean magnitudes of the original signal.
    std_magnitudes (numpy.array): Standard deviations of the magnitudes.
    mean_phases (numpy.array): Mean phases of the original signal.
    std_phases (numpy.array): Standard deviations of the phases.
    state (str): The state of the signal ('normal' or 'abnormal').
    args (PassArgs): Additional arguments for the function.
    size (int): The size of the generated signal.
    discountFactor (float): A factor to discount the size of the signal.

    Returns:
    numpy.array: The generated signal.
    """
    if args is None:
        args = PassArgs()

    if np.random.rand() < 0.5:
        cycleTime = np.random.rand() * args.RescalFactor * size / 40 + 3
    else:
        cycleTime = np.random.rand() * args.RescalFactor * (size / 40 - 10) + 10

    if state != 'abnormal':
        cycleTime = 1

    sizeOr = copy.deepcopy(size)
    size = int(size / cycleTime)
    GenerFlag = True

    while GenerFlag:
        low = 0.01
        if discountFactor is None:
            if np.random.rand() < 0.7 and state == 'abnormal':
                discountFactor = np.exp(np.random.uniform(low=np.log(low), high=np.log(15)))
            else:
                discountFactor = np.exp(np.random.uniform(low=np.log(low * 50), high=np.log(15)))


        discountSize = int(size / discountFactor)

        random_magnitudes = np.random.normal(loc=mean_magnitudes, scale=std_magnitudes * 5)
        random_phases = np.random.normal(loc=mean_phases, scale=std_phases * 5)

        if np.random.rand() < 0.5:
            random_phases = -random_phases
        if np.random.rand() < 0.5:
            random_magnitudes = -random_magnitudes

        new_fft_signal = random_magnitudes * np.exp(1j * random_phases)
        new_fft_signal = np.real(np.fft.ifft(new_fft_signal)).flatten()

        fft_result = np.fft.fft(new_fft_signal)
        random_magnitudes = np.abs(fft_result)
        random_phases = np.angle(fft_result)
        random_magnitudes = extend_array_to_target_length(random_magnitudes, discountSize)
        random_magnitudes *= len(random_magnitudes) / len(mean_magnitudes)
        random_phases = extend_array_to_target_length(random_phases, discountSize)
        new_fft_signal2 = random_magnitudes * np.exp(1j * random_phases)
        new_fft_signal2 = np.real(np.fft.ifft(new_fft_signal2)).flatten()[:discountSize]

        t_orig = np.linspace(0, 1, discountSize)
        t_new = np.linspace(0, 1, size)

        new_fft_signal2 = np.interp(t_new, t_orig, new_fft_signal2)


        p = np.random.rand()
        if p < 0.1 and cycleTime > 5:
            new_fft_signal2 = random_walk(size)
            k = np.random.rand() * 20 + 1
            window = np.random.randint(20, 100)
        elif p < 0.2 and cycleTime > 5:
            new_fft_signal2 = cyclic_pattern(size)
            k = np.random.rand() * 20 + 1
            window = np.random.randint(20, 100)
        else:
            k = 1
            window = 10

        new_length = int(k * size)
        new_fft_signal2 = np.interp(np.linspace(0, size - 1, new_length), np.arange(size), new_fft_signal2)[:size]

        if np.random.rand() < 0.3 and state == 'abnormal':
            emd = EMD()
            IMFs = emd.emd(new_fft_signal2, np.linspace(0, 1, len(new_fft_signal2)))
            p = np.random.rand()
            if p < 0.15:
                try:
                    residue = IMFs[-3] + IMFs[-2] + IMFs[-1]
                except:
                    try:
                        residue = IMFs[-2] + IMFs[-1]
                    except:
                        residue = IMFs[-1]
            elif p < 0.3:
                try:
                    residue = IMFs[-2] + IMFs[-1]
                except:
                    residue = IMFs[-1]
            else:
                residue = IMFs[-1]
            residue -= residue.mean()
        else:
            residue = 0
        new_fft_signal2 -= residue
        if (new_fft_signal2.max() - new_fft_signal2.min()) > 1:
            GenerFlag = False



        if args.RescalFactor == 1 and np.random.rand() < 0.5 and discountFactor > 2:
            start = int(np.random.rand() * size) + window + 1
            new_fft_signal3 = np.tile(new_fft_signal2, int((cycleTime + 20)) * 2)
            new_fft_signal3 = smooth_signal(new_fft_signal3, window_size=int(window) + 1)[start:start + sizeOr * 2]
            new_fft_signal2 = (new_fft_signal3[0::2] + new_fft_signal3[1::2]) / 2
        else:
            start = int(np.random.rand() * size) + window + 1
            new_fft_signal2 = np.tile(new_fft_signal2, int((cycleTime * args.RescalFactor + 20)))
            new_fft_signal2 = smooth_signal(new_fft_signal2, window_size=int(window) + 1)[start:start + sizeOr]

    new_fft_signal2 = (new_fft_signal2 - new_fft_signal2.min()) / (new_fft_signal2.max() - new_fft_signal2.min())

    p = np.random.rand()
    if state == 'Type1' and p < 0.4:
        t = np.linspace(0, 1, sizeOr, endpoint=False)
        frequency = np.random.randint(3, int(sizeOr / 1800 * 20))
        k = np.random.rand() * 7 + 1
        wave = np.sin(2 * np.pi * frequency * t) + 1
        d = np.random.rand() * 0.8 + 1
        wave[wave < d] = 1
        wave[wave > 1] = wave[wave > 1] * k
        new_fft_signal2 = new_fft_signal2 * wave
        if args.RescalFactor == 1 and np.random.rand() < 0.5:
            start = int(np.random.rand() * size) + 3 + 1
            new_fft_signal3 = np.tile(new_fft_signal2, 4)
            new_fft_signal3 = smooth_signal(new_fft_signal3, window_size=int(3) + 1)[start:start + sizeOr * 2]
            new_fft_signal2 = (new_fft_signal3[0::2] + new_fft_signal3[1::2]) / 2
    elif state == 'Type1' and p < 0.7:
        t = np.linspace(0, 1, sizeOr, endpoint=False)
        frequency = np.random.randint(3, int(sizeOr / 1800 * 10))
        k = np.random.rand() * 1 + 1
        wave = np.sin(2 * np.pi * frequency * t) + 1
        d = np.random.rand() * 0.8 + 1
        wave[wave < d] = 1
        wave[wave > 1] = wave[wave > 1] * k
        new_fft_signal2 = new_fft_signal2 * wave

    return new_fft_signal2


def RWB(size, start=0):
    """
    Generate a random walk signal.

    Parameters:
    size (int): The length of the generated signal.
    start (float): The starting value of the random walk.

    Returns:
    numpy.array: The generated random walk signal.
    """
    steps = np.random.normal(size=size)
    if np.random.rand() < 0.5:
        return np.cumsum(steps) + start
    else:
        return np.cumsum(steps) + start + cyclic_pattern(size)


def seasonal_periodicity(size, num_periods=2):
    """
    Generate a signal with seasonal periodicity.

    Parameters:
    size (int): The length of the generated signal.
    num_periods (int): The number of periods in the signal.

    Returns:
    numpy.array: The generated signal with seasonal periodicity.
    """
    x = np.arange(size)
    components = []
    num_periods = np.random.randint(low=1, high=5)
    for _ in range(num_periods):
        amplitude = np.random.uniform(0.5, 2)
        period = np.random.uniform(low=20, high=size * 3)
        phase_shift = np.random.randint(0, period)
        components.append(amplitude * np.sin((2 * np.pi / period) * (x + phase_shift)))
    if np.random.rand() < 0.5:
        return np.sum(components, axis=0) + np.random.normal(size=size)
    else:
        return np.sum(components, axis=0) + np.random.normal(size=size) + cyclic_pattern(size)

def triangle_wave(size, cycle_length, amplitude,decay=1):
    x = np.linspace(0, 1, int(cycle_length / 2))
    first_half = amplitude * x
    second_half = amplitude * (1 - x)
    single_cycle = np.concatenate((first_half, second_half))
    return np.tile(single_cycle, size // len(single_cycle) + 1)[:size]*decay


def square_wave(size, cycle_length, amplitude):
    single_cycle = np.concatenate(
        (np.full(int(cycle_length / 2), amplitude), np.full(int(cycle_length / 2), -amplitude)))
    return np.tile(single_cycle, size // len(single_cycle) + 1)[:size]


def step_wave(size, cycle_length, amplitude):
    single_cycle = np.concatenate((np.full(int(cycle_length / 2), amplitude), np.full(int(cycle_length / 2), 0)))
    return np.tile(single_cycle, size // len(single_cycle) + 1)[:size]

def sawtooth_wave(size, cycle_length, amplitude):
    x = np.linspace(0, 1, int(cycle_length))
    single_cycle = amplitude * x
    return np.tile(single_cycle, size // len(single_cycle) + 1)[:size]

def cyclic_pattern_base(size, wave_general_type='other'):
    """
    Generate a base cyclic pattern signal.

    Parameters:
    size (int): The length of the generated signal.
    wave_general_type (str): The general type of wave ('cycle' or other).

    Returns:
    numpy.array: The generated base cyclic pattern signal.
    """
    x = np.arange(size)
    s = size if size < 1000 else 1000

    cycle_length = np.exp(np.random.uniform(low=np.log(11), high=np.log(s * 2)))
    amplitude = np.random.uniform(0.5, 2)

    if wave_general_type == 'cycle':
        wave_type = np.random.choice(['sine', 'triangle', 'square', 'step'])
    else:
        wave_type = np.random.choice(['sine'])

    if wave_type == 'sine':
        wave = amplitude * np.sin((2 * np.pi / cycle_length) * x)
    elif wave_type == 'triangle':
        wave = triangle_wave(size, cycle_length, amplitude)
    elif wave_type == 'square':
        wave = square_wave(size, cycle_length, amplitude)
    elif wave_type == 'step':
        wave = step_wave(size, cycle_length, amplitude)
    elif wave_type == 'sawtooth':
        wave = sawtooth_wave(size, cycle_length, amplitude)

    return wave + amplitude / 10 * np.random.normal(size=size)


def PWB(size, cycle_number=None, wave_type="cycle"):
    """
    Generate a signal with a cyclic pattern.

    Parameters:
    size (int): The length of the generated signal.
    cycle_number (int): The number of cycles in the signal.
    wave_type (str): The type of wave ('cycle' or other).

    Returns:
    numpy.array: The generated signal with a cyclic pattern.
    """
    if cycle_number is None:
        cycle_number = np.random.randint(1, 8)
    wave = cyclic_pattern_base(size, wave_general_type="cycle")
    for _ in range(cycle_number - 1):
        wave += cyclic_pattern_base(size, wave_general_type=wave_type)
    return wave


def TWDB(size):
    """
    Generate a trending signal.

    Parameters:
    size (int): The length of the generated signal.

    Returns:
    numpy.array: The generated trending signal.
    """
    start = np.random.uniform(-10, 10)
    trend = np.random.uniform(-1, 1)
    if np.random.rand() < 0.1:
        return start + trend * np.arange(size) + 0.5 * np.random.normal(size=size) + np.random.normal(size=size)
    else:
        return (start + trend * np.arange(size) + 0.5 * np.random.normal(size=size) + np.random.normal(size=size)) * 0.005 + cyclic_pattern(size)


def LGB(size):
    """
    Generate a logistic growth signal.

    Parameters:
    size (int): The length of the generated signal.

    Returns:
    numpy.array: The generated logistic growth signal.
    """
    saturation = np.exp(np.random.uniform(low=np.log(1), high=np.log(10)))
    growth_rate = np.exp(np.random.uniform(low=np.log(0.001), high=np.log(0.1)))
    start = np.random.uniform(0, size * 0.9)
    x = np.arange(size)
    if np.random.rand() < 0.5:
        return saturation / (1 + np.exp(-growth_rate * (x - start))) + 0.05 * np.random.normal(size=size)
    else:
        return saturation / (1 + np.exp(-growth_rate * (x - start))) + 0.05 * np.random.normal(size=size) + cyclic_pattern(size)


def amplify_data(arr):
    """
    Amplify a portion of the data by a random factor.

    Parameters:
    arr (numpy.array): The input data array.

    Returns:
    numpy.array: The amplified data array.
    """
    return_arr = np.copy(arr)
    percent = np.random.rand() * 0.1 + 0.15
    scalar = np.random.rand() * 5 + 1.5
    top_10_percent = int(percent * len(arr))

    if top_10_percent == 0:
        return return_arr

    factor = 1 if np.random.rand() > 0.5 else -1
    indices = np.argsort(-factor * arr)[:top_10_percent]

    return_arr[indices] *= scalar

    return return_arr

if __name__ =='__main__':
    import matplotlib.pyplot as plt
    import pickle
    with open('IFFTB_Distribution1', 'rb') as f:
        mean_magnitudes1, std_magnitudes1, mean_phases1, std_phases1 = pickle.load(f)
    with open('IFFTB_Distribution2', 'rb') as f:
        mean_magnitudes2, std_magnitudes2, mean_phases2, std_phases2 = pickle.load(f)



    for i in range(100):

        d1=IFFTB( mean_magnitudes1, std_magnitudes1, mean_phases1, std_phases1,'Type1',size=3500,discountFactor=None)
        # d2=IFFTB( mean_magnitudes2, std_magnitudes2, mean_phases2, std_phases2,'Type2',size=2500,discountFactor=None)
        plt.figure(figsize=(10, 1))
        plt.plot(d1)
        # plt.plot(d2)
        plt.show()