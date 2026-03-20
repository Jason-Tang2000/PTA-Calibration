# This is a modification (Matlab to python) from Vaclavek's code.
# Joanna 2025/02/24

import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QTextEdit
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import datetime
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d, PchipInterpolator
from PyQt5.QtGui import QFont

## Parameters for the microphone
device = 24
chan_out_L = 5 
chan_out_R = 8

class AudiometerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.sound_playing = False
        self.hit_key = False
        self.setFocusPolicy(Qt.StrongFocus)  # Ensure the main window captures key events
        self.scatter_handles = {'Left Ear': {}, 'Right Ear': {}}
    
    def initUI(self):
        self.setWindowTitle('Audiometer')
        self.setGeometry(150, 250, 950, 400)
        
        # Create a QFont object with the desired font size
        font = QFont()
        font.setPointSize(11)
        font.setFamily('Arial')
        
        # Graph setup
        self.figure, self.ax = plt.subplots()
        self.ax.set_xscale('log')
        self.ax.set_xlim([125, 16000])
        self.ax.set_ylim([-20, 120])
        self.ax.axhline(0, color='black', linewidth=1.5)
        self.ax.axhline(y=20, color='black', linewidth=0.8, alpha=0.5)
        self.ax.invert_yaxis()
        xticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        self.ax.set_xticks(xticks, labels=[str(x) for x in xticks])
        self.ax.set_xlabel('Frequency (Hz)', size=16)
        self.ax.set_ylabel('Level (dB re abs. threshold)', size=16)
        self.ax.tick_params(axis='both', which='major', labelsize=16)
        self.ax.grid(visible=True, alpha=0.3)
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(900, 700)
        
        # UI Elements
        self.subject_id_label = QLabel('Subject ID')
        self.subject_id_label.setFont(font)
        self.subject_id_input = QLineEdit()
        self.subject_id_input.setFixedWidth(150)
        self.subject_id_input.setFont(font)

        self.tone_select = QComboBox()
        self.tone_select.addItems(["Pure Tone", "Warble Tone (freq)", "Warble Tone (amp)"])
        self.tone_select.setFixedWidth(210)
        self.tone_select.setFixedHeight(40)
        self.tone_select.setFont(font)

        self.ear_select = QComboBox()
        self.ear_select.addItems(["Left Ear", "Right Ear"])
        self.ear_select.setFixedWidth(150)
        self.ear_select.setFixedHeight(40)
        self.ear_select.setFont(font)

        self.freq_label = QLabel('Frequency (Hz)')
        self.freq_label.setFont(font)
        self.freq_input = QLineEdit('1000')
        self.freq_input.setFixedWidth(150)
        self.freq_input.setFixedHeight(40)
        self.freq_input.setAlignment(Qt.AlignCenter)
        self.freq_input.setFont(font)

        self.low_button = QPushButton('Low')
        self.low_button.clicked.connect(self.decrease_freq)
        self.low_button.setFixedWidth(100)
        self.low_button.setFixedHeight(35)
        self.low_button.setFont(font)

        self.high_button = QPushButton('High')
        self.high_button.clicked.connect(self.increase_freq)
        self.high_button.setFixedWidth(100)
        self.high_button.setFixedHeight(35)
        self.high_button.setFont(font)
        
        self.level_label = QLabel('Level (dB re abs. threshold)')
        self.level_label.setFont(font)
        self.level_input = QLineEdit('50')
        self.level_input.setFixedWidth(150)
        self.level_input.setFixedHeight(40)
        self.level_input.setAlignment(Qt.AlignCenter)
        self.level_input.setFont(font)
        
        self.down_button = QPushButton('Down')
        self.down_button.clicked.connect(self.decrease_level)
        self.down_button.setFixedWidth(100)
        self.down_button.setFixedHeight(35)
        self.down_button.setFont(font)
        
        self.up_button = QPushButton('Up')
        self.up_button.clicked.connect(self.increase_level)
        self.up_button.setFixedWidth(100)
        self.up_button.setFixedHeight(35)
        self.up_button.setFont(font)

        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.play_sound)
        self.play_button.setFixedWidth(160)
        self.play_button.setFixedHeight(60)
        self.play_button.setFont(font)
        
        self.save_button = QPushButton('Save Result')
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setFixedWidth(160)
        self.save_button.setFixedHeight(60)
        self.save_button.setFont(font)

        self.freq_values = {}
        self.freq_values['Left Ear'] = []
        self.freq_values['Right Ear'] = []
        self.level_values = {}
        self.level_values['Left Ear'] = []
        self.level_values['Right Ear'] = []

        self.save_all_button = QPushButton('Save All')
        self.save_all_button.clicked.connect(self.save_all_data)
        self.save_all_button.setFixedWidth(170)
        self.save_all_button.setFixedHeight(80)
        self.save_all_button.setFont(font)

        # need to change to the corresponding file for different headphones
        self.leftInfo = QLabel('Left Ear')
        self.leftInfo.setFont(font)
        # self.leftCalib = QLineEdit('HDA-300left01.mat')
        # self.leftCalib = QLineEdit('DD-45left01-SPL.mat')
        self.leftCalib = QLineEdit('DD-45left-rms.mat')
        self.leftCalib.setFixedWidth(200)
        self.leftCalib.setFont(font)
        # self.RETSPLleft = QLineEdit('RETSPLhda300.mat')
        self.RETSPLleft = QLineEdit('RETSPLdd45.mat')
        self.RETSPLleft.setFixedWidth(200)
        self.RETSPLleft.setFont(font)
        self.rightInfo = QLabel('Right Ear')
        self.rightInfo.setFont(font)
        # self.rightCalib = QLineEdit('HDA-300right01.mat')
        # self.rightCalib = QLineEdit('DD-45right01-SPL.mat')
        self.rightCalib = QLineEdit('DD-45right-rms.mat')
        self.rightCalib.setFixedWidth(200)
        self.rightCalib.setFont(font)
        # self.RETSPLright = QLineEdit('RETSPLhda300.mat')
        self.RETSPLright = QLineEdit('RETSPLdd45.mat')
        self.RETSPLright.setFixedWidth(200)
        self.RETSPLright.setFont(font)

        # the indicator (a red/green board) of whether the listener hit the keyboard
        # (origin: grey, red: not hit, green: hit)
        self.hit = QLabel()
        self.hit.setStyleSheet("background-color: grey")
        self.hit.setFixedHeight(60)
        self.hit.setFixedWidth(420)
        self.hit.setFont(font)
        
        self.feedback_label = QTextEdit()
        self.feedback_label.setStyleSheet("color: black")
        self.feedback_label.setFixedWidth(420)
        self.feedback_label.setReadOnly(True)
        self.feedback_label.setFont(font)

        self.print_results = QPushButton('Print Final Results')
        self.print_results.clicked.connect(self.print_results_graph)
        self.print_results.setFixedWidth(200)
        self.print_results.setFixedHeight(70)
        self.print_results.setFont(font)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setFixedWidth(150)
        self.reset_button.setFixedHeight(80)
        self.reset_button.setFont(font)

        # Layouts
        vbox1 = QVBoxLayout()
        vbox1_1 = QVBoxLayout()
        vbox1_1.addWidget(self.leftInfo)
        vbox1_1.addWidget(self.leftCalib)
        vbox1_1.addWidget(self.RETSPLleft)
        vbox1_2 = QVBoxLayout()
        vbox1_2.addWidget(self.rightInfo)
        vbox1_2.addWidget(self.rightCalib)
        vbox1_2.addWidget(self.RETSPLright)
        hbox3 = QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addLayout(vbox1_1)
        hbox3.addLayout(vbox1_2)
        hbox3.addStretch(1)
        vbox1.addLayout(hbox3)
        vbox1.addStretch(1)

        hbox4 = QHBoxLayout()
        hbox4.addStretch(1)
        hbox4.addWidget(self.subject_id_label)
        hbox4.addWidget(self.subject_id_input)
        hbox4.addStretch(1)
        vbox1.addLayout(hbox4)
        
        vbox1.addStretch(1)
        
        hbox_ear_select = QHBoxLayout()
        hbox_ear_select.addStretch(1)
        hbox_ear_select.addWidget(self.tone_select)
        hbox_ear_select.addStretch(1)
        hbox_ear_select.addWidget(self.ear_select)
        hbox_ear_select.addStretch(1)
        vbox1.addLayout(hbox_ear_select)

        vbox1.addStretch(1)

        hbox_freq_label = QHBoxLayout()
        hbox_freq_label.addStretch(1)
        hbox_freq_label.addWidget(self.freq_label)
        hbox_freq_label.addStretch(1)
        vbox1.addLayout(hbox_freq_label)

        hbox_freq_input = QHBoxLayout()
        hbox_freq_input.addStretch(1)
        hbox_freq_input.addWidget(self.freq_input)
        hbox_freq_input.addStretch(1)
        vbox1.addLayout(hbox_freq_input)

        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(self.low_button)
        hbox2.addWidget(self.high_button)
        hbox2.addStretch(1)
        vbox1.addLayout(hbox2)

        vbox1.addStretch(1)

        hbox_level_label = QHBoxLayout()
        hbox_level_label.addStretch(1)
        hbox_level_label.addWidget(self.level_label)
        hbox_level_label.addStretch(1)
        vbox1.addLayout(hbox_level_label)

        hbox_level_input = QHBoxLayout()
        hbox_level_input.addStretch(1)
        hbox_level_input.addWidget(self.level_input)
        hbox_level_input.addStretch(1)
        vbox1.addLayout(hbox_level_input)

        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(self.down_button)
        hbox1.addWidget(self.up_button)
        hbox1.addStretch(1)
        vbox1.addLayout(hbox1)

        vbox1.addStretch(1)

        hbox_play_button = QHBoxLayout()
        hbox_play_button.addStretch(1)
        hbox_play_button.addWidget(self.play_button)
        hbox_play_button.addStretch(1)
        vbox1.addLayout(hbox_play_button)

        vbox1.addStretch(1)

        hbox_save_button = QHBoxLayout()
        hbox_save_button.addStretch(1)
        hbox_save_button.addWidget(self.save_button)
        hbox_save_button.addStretch(1)
        vbox1.addLayout(hbox_save_button)
        
        vbox3 = QVBoxLayout()

        hbox_print_results = QHBoxLayout()
        hbox_print_results.addStretch(1)
        hbox_print_results.addWidget(self.print_results)
        hbox_print_results.addStretch(1)
        vbox3.addLayout(hbox_print_results)

        vbox3.addStretch(1)

        hbox_hit = QHBoxLayout()
        hbox_hit.addStretch(1)
        hbox_hit.addWidget(self.hit)
        hbox_hit.addStretch(1)
        vbox3.addLayout(hbox_hit)

        hbox_feedback_label = QHBoxLayout()
        hbox_feedback_label.addStretch(1)
        hbox_feedback_label.addWidget(self.feedback_label)
        hbox_feedback_label.addStretch(1)
        vbox3.addLayout(hbox_feedback_label)

        vbox3.addStretch(1)

        hbox_save_all_button = QHBoxLayout()
        hbox_save_all_button.addStretch(1)
        hbox_save_all_button.addWidget(self.save_all_button)
        hbox_save_all_button.addStretch(1)
        vbox3.addLayout(hbox_save_all_button)

        vbox3.addStretch(1)

        hbox_reset_button = QHBoxLayout()
        hbox_reset_button.addStretch(1)
        hbox_reset_button.addWidget(self.reset_button)
        hbox_reset_button.addStretch(1)
        vbox3.addLayout(hbox_reset_button)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(vbox1)
        main_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        main_layout.addLayout(vbox3)
        
        self.setLayout(main_layout)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space: # and self.sound_playing:
            self.hit_key = True
            self.hit.setStyleSheet("background-color: green")
            ear_tmp = self.ear_select.currentText()
            if ear_tmp == 'Left Ear':
                ear = "L"
            else:
                ear = "R"
            self.feedback_label.append(f'<span style="color:green">{ear}: Response at {self.freq_input.text()} Hz and {self.level_input.text()} dB</span>')
            # sd.stop()
            # self.sound_playing = False

    def play_sound(self):
        fs = 48000  # Sample rate
        duration = 2  # seconds
        freq = float(self.freq_input.text())
        level = float(self.level_input.text())
        stimulus = self.tone_select.currentText()

        ear = self.ear_select.currentText()
        if ear == 'Left Ear':
            chan_out = chan_out_L
        else:
            chan_out = chan_out_R
        
        # Generate tone
        tx = np.arange(0, duration, 1/fs)
        # Generate tone
        tx = np.arange(0, duration, 1/fs)
        if stimulus == "Pure Tone":
            tone = np.sin(2 * np.pi * freq * tx)
        elif stimulus == "Warble Tone (freq)":
            ## frequency modulated tone
            freq_dev = 0.025 * freq  # ±2.5%
            mod_freq = 10  # Hz modulation frequency

            inst_freq = freq + freq_dev * np.sin(2 * np.pi * mod_freq * tx)
            phase = 2 * np.pi * np.cumsum(inst_freq) / fs
            tone = np.sin(phase)
        elif stimulus == "Warble Tone (amp)":
            ## Amplitude modulated tone
            mod_freq = 3  # Hz modulation frequency
            mod_depth = 0.5  # modulation depth (0 to 1)
            mod_signal = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * tx)
            tone = np.sin(2 * np.pi * freq * tx) * mod_signal

        # Ramping
        ramp_dur = 50e-3  # 50 ms
        x = np.arange(0, ramp_dur, 1/fs)
        ramp_up = 0.5 * (1 - np.cos(np.pi * x / ramp_dur))
        ramp_down = ramp_up[::-1]
        whole_ramp = np.concatenate((ramp_up, np.ones(len(tone) - 2 * len(x)), ramp_down))
        tone *= whole_ramp

        if ear == 'Left Ear':
            calib_file = self.leftCalib.text()
            RETSPL_file = self.RETSPLleft.text()
            ATfreq = loadmat(RETSPL_file)['ATfreq'].squeeze()
            AThresh = loadmat(RETSPL_file)['AThresh'].squeeze()
            data_tmp = loadmat(calib_file)['data'][0, 0]
            # Convert to a dictionary
            data = {name: data_tmp[name] for name in data_tmp.dtype.names}
        else :
            calib_file = self.rightCalib.text()
            RETSPL_file = self.RETSPLright.text()
            ATfreq = loadmat(RETSPL_file)['ATfreq'].squeeze()
            AThresh = loadmat(RETSPL_file)['AThresh'].squeeze()
            data_tmp = loadmat(calib_file)['data'][0, 0]
            # Convert to a dictionary
            data = {name: data_tmp[name] for name in data_tmp.dtype.names}
        
        # Interpolation of RETSPL
        # fxI = np.arange(1, 20001, 1)
        # AThreshI = interp1d(ATfreq, AThresh, kind='cubic', fill_value='extrapolate')(fxI)

        # Transfer function
        fxA = data['fx'].squeeze()
        HresS = np.abs(data['TrFuncAE'].squeeze())

        # fx = np.arange(100, 15001, 1)
        # interp_func = PchipInterpolator(fxA, HresS)  # Create PCHIP interpolator
        # Lx1H = interp_func(fx)  # Interpolate at points fx
        
        # Compute level adjustments
        idxF = np.where(fxA == freq)[0][0]
        idxRETSPL = np.where(ATfreq == freq)[0][0]
        levelSPL = level + AThresh[idxRETSPL]   ## ***here need to change!!!!***
        scale1 = 10**(levelSPL / 20) * np.sqrt(2) * 2e-5 / HresS[idxF]

        signal = tone * scale1  # Apply scaling
        
        self.sound_playing = True
        self.hit_key = False
        self.hit.setStyleSheet("background-color: grey")
        sd.play(signal, fs, device=device, mapping=chan_out)
        
        QTimer.singleShot(duration * 1000, self.check_hit)
        
        self.setFocus()  # Set focus back to the main window

    def check_hit(self):
        if not self.hit_key:
            self.hit.setStyleSheet("background-color: red")
            ear_tmp = self.ear_select.currentText()
            if ear_tmp == 'Left Ear':
                ear = "L"
            else:
                ear = "R"
            self.feedback_label.append(f'<span style="color:red">{ear}: No response at {self.freq_input.text()} Hz and {self.level_input.text()} dB</span>')
        self.sound_playing = False

    def decrease_freq(self):
        freq = int(self.freq_input.text())
        freq_list = [250, 500, 1000, 2000, 4000, 6000, 8000]
        if freq in freq_list:
            freq_index = freq_list.index(freq)
        else:
            freq_index = np.argmin(np.abs(np.array(freq_list) - freq))
            # print(f"Warning: Frequency {freq} not in list, using closest value {freq_list[freq_index]}")

        if freq_index > 0 and freq <= freq_list[freq_index]:
            self.freq_input.setText(str(freq_list[freq_index - 1]))
        elif freq_index > 0 and freq > freq_list[freq_index]:
            self.freq_input.setText(str(freq_list[freq_index]))
        elif freq_index == len(freq_list) - 1:
            self.freq_input.setText(str(freq_list[freq_index]))

    def increase_freq(self):
        freq = int(self.freq_input.text())
        freq_list = [250, 500, 1000, 2000, 4000, 6000, 8000]
        if freq in freq_list:
            freq_index = freq_list.index(freq)
        else:
            freq_index = np.argmin(np.abs(np.array(freq_list) - freq))
            # print(f"Warning: Frequency {freq} not in list, using closest value {freq_list[freq_index]}")

        if freq_index < len(freq_list) - 1:
            self.freq_input.setText(str(freq_list[freq_index + 1]))
        elif freq_index == 0:
            self.freq_input.setText(str(freq_list[freq_index]))

    def decrease_level(self):
        level = int(self.level_input.text())
        if level > -20:
            self.level_input.setText(str(level - 5))
    
    def increase_level(self):
        level = int(self.level_input.text())
        if level < 120:
            self.level_input.setText(str(level + 5))
    
    def save_data(self):
        # freq = int(self.freq_input.text())
        # level = float(self.level_input.text())
        # ear = self.ear_select.currentText()

        # self.freq_values[ear].append(freq)
        # self.level_values[ear].append(level)
        
        # if ear == 'Left Ear':
        #     self.ax.scatter(freq, level, marker='x', color='b', label='Left Ear', s=100)
        # else:
        #     self.ax.scatter(freq, level, marker='o', color='r', label='Right Ear', s=100)

        # self.canvas.draw()

        freq = int(self.freq_input.text())
        level = float(self.level_input.text())
        ear = self.ear_select.currentText()

        # Check if a previous scatter exists at this freq → remove it
        if freq in self.scatter_handles[ear]:
            old_scatter = self.scatter_handles[ear][freq]
            old_scatter.remove()  # Remove the artist from the plot
            # also remove the point in freq_values and level_values
            index = self.freq_values[ear].index(freq)
            del self.freq_values[ear][index]
            del self.level_values[ear][index]

        self.freq_values[ear].append(freq)
        self.level_values[ear].append(level)

        # Plot new scatter and save the handle
        if ear == 'Left Ear':
            scatter = self.ax.scatter(freq, level, marker='x', color='b', 
                                      label='Left Ear', linewidths=2, s=100)
        else:
            scatter = self.ax.scatter(freq, level, marker='o', facecolors='none', color='r', 
                                      label='Right Ear', linewidths=2, s=120)

        # Store the new scatter handle
        self.scatter_handles[ear][freq] = scatter

        self.canvas.draw()
    
    def print_results_graph(self):
        # first sort the data base on freq_values, then sort the level_values
        # the sorted freq_values and level_values will be matched

        self.ax.clear()
        self.ax.set_xscale('log')
        self.ax.set_xlim([125, 16000])
        self.ax.set_ylim([-20, 120])
        self.ax.axhline(0, color='black', linewidth=1.5)
        self.ax.axhline(y=20, color='black', linewidth=0.8, alpha=0.5)
        self.ax.invert_yaxis()
        xticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        self.ax.set_xticks(xticks, labels=[str(x) for x in xticks])

        self.ax.set_xlabel('Frequency (Hz)', size=16)
        self.ax.set_ylabel('Level (dB re abs. threshold)', size=16)
        self.ax.tick_params(axis='both', which='major', labelsize=16)
        self.ax.grid(visible=True, alpha=0.3)
        
        for ear in ['Left Ear', 'Right Ear']:
            if len(self.freq_values[ear]) > 0:
                self.freq_values[ear], self.level_values[ear] = zip(*sorted(zip(self.freq_values[ear], self.level_values[ear])))

                if ear == 'Left Ear':
                    self.ax.plot(self.freq_values[ear], self.level_values[ear], 'bx-', markersize=10,
                                label='Left Ear', linewidth=2, markeredgewidth=2)
                else:
                    self.ax.plot(self.freq_values[ear], self.level_values[ear], 'ro-', markerfacecolor='none',
                                markersize=11, label='Right Ear', linewidth=2, markeredgewidth=2)
        self.ax.legend()
        self.canvas.draw()

    def save_all_data(self):
        # Save the data to a file
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        subject_id = self.subject_id_input.text()
        filename = f'results\{subject_id}_PTA_{now}.txt'
        # create file if not exist
        if not os.path.exists('results'):
            os.makedirs('results')
        
        with open(filename, 'w') as f:
            f.write(f"Subject ID: {subject_id}\n")
            f.write(f"Date: {now}\n")
            f.write("Ear, Frequency (Hz), Level (dB re abs. threshold)\n")
            for ear in ['Left Ear', 'Right Ear']:
                for f_val, l_val in zip(self.freq_values[ear], self.level_values[ear]):
                    f.write(f"{ear}, {f_val}, {l_val}\n")
        
        # also save the graph
        graph_filename = f'results\{subject_id}_PTA_{now}.png'
        self.figure.savefig(graph_filename)

        print(f"Results saved to {filename}")
    
    def reset(self):
        self.freq_values = {}
        self.freq_values['Left Ear'] = []
        self.freq_values['Right Ear'] = []
        self.level_values = {}
        self.level_values['Left Ear'] = []
        self.level_values['Right Ear'] = []
        self.scatter_handles = {'Left Ear': {}, 'Right Ear': {}}

        self.ax.clear()
        self.ax.set_xscale('log')
        self.ax.set_xlim([125, 16000])
        self.ax.set_ylim([-20, 120])
        self.ax.axhline(0, color='black', linewidth=1.5)
        self.ax.axhline(y=20, color='black', linewidth=0.8, alpha=0.5)
        self.ax.invert_yaxis()
        xticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        self.ax.set_xticks(xticks, labels=[str(x) for x in xticks])
        self.ax.set_xlabel('Frequency (Hz)', size=16)
        self.ax.set_ylabel('Level (dB re abs. threshold)', size=16)
        self.ax.tick_params(axis='both', which='major', labelsize=16)
        self.ax.grid(visible=True, alpha=0.3)
        self.canvas.draw()
        
        self.feedback_label.clear()
        self.hit.setStyleSheet("background-color: grey")

        self.freq_input.setText('1000')
        self.level_input.setText('50')
        self.subject_id_input.setText('')
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudiometerApp()
    ex.show()
    sys.exit(app.exec_())
