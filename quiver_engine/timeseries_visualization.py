# Copyright 2017 Patrick Bos, Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib as mpl
mpl.use('agg')
print(mpl.get_backend())
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_timeseries(timeseries, dt=1., ax=None, minimal_layout=True,
                    draw_labels=True, title=None, draw_colorbar=False,
                    center_zero_color=True, base_height=1., channel_px=5):
    """
    @param timeseries: numpy array of shape (T, C) (T timebins by C channels).
    @param dt: time step (in seconds) between timeseries bins (used for tick units).
    """
    N_timebins, N_channels = timeseries.shape

    if ax is None:
        height = base_height
        width = base_height * (N_timebins / (N_channels * channel_px))
        figsize = (width, height)
        if minimal_layout:
            fig, ax = plt.subplots(1, frameon=False, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.figure

    if not minimal_layout and draw_labels:
        ax.set_xlabel('time [s]')
        ax.set_ylabel('channel')

    if minimal_layout:
        # ax.axis('tight')
        ax.axis('off')

    v_extreme = max(timeseries.max(), abs(timeseries.min()))

    im = ax.imshow(timeseries.T, aspect='auto', interpolation='nearest', cmap=cm.RdYlBu,
                   extent=(0, N_timebins * dt, N_channels, 0), vmin=-v_extreme, vmax=v_extreme)

    if not minimal_layout and title is not None:
        ax.set_title(title)

    if draw_colorbar:
        cb = fig.colorbar(im, ax=ax)
    else:
        cb = None

    # plt.tight_layout()

    return {'ax': ax, 'fig': fig, 'im': im, 'cb': cb}


def generate_timeseries_image(input_data, fn_png, **kwargs):
    plot_results = plot_timeseries(input_data, **kwargs)
    plot_results['fig'].savefig(fn_png, bbox_inches='tight', pad_inches=0)
    plt.close(plot_results['fig'])


def plot_timeseries_fourier_amplitudes(timeseries, dt, ax=None, minimal_layout=True,
                                       draw_labels=True, title=None, draw_colorbar=False):
    """
    @param timeseries: numpy array of shape (T, C) (T timebins by C channels).
    @param dt: time step (in seconds) between timeseries bins.
    """
    if ax is None:
        if minimal_layout:
            fig, ax = plt.subplots(1, frameon=False)
        else:
            fig, ax = plt.subplots(1)
    else:
        fig = ax.figure

    if not minimal_layout and draw_labels:
        ax.set_xlabel('frequency [s^{-1}]')
        ax.set_ylabel('channel')

    N_timebins, N_channels = timeseries.shape

    frequencies = np.fft.fftfreq(N_timebins, d=dt)  # s^{-1}
    central_timebin = N_timebins // 2
    if N_timebins % 2 == 0:
        central_timebin -= 1
    fmin = frequencies[0]
    fmax = frequencies[central_timebin]

    fourier_transform = np.fft.fft(timeseries, axis=0)[:central_timebin + 1]

    fourier_amplitudes = np.abs(fourier_transform)

    im = ax.imshow(fourier_amplitudes.T, aspect='auto', interpolation='nearest',
                   extent=(fmin, fmax, N_channels, 0))

    if not minimal_layout and title is not None:
        ax.set_title(title)

    if draw_colorbar:
        cb = fig.colorbar(im, ax=ax)
    else:
        cb = None

    if minimal_layout:
        ax.axis('off')

    plt.tight_layout()

    return {'ax': ax, 'fig': fig, 'im': im, 'cb': cb}
