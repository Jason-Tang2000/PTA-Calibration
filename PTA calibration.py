# %%
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
# import pandas as pd
from scipy.io import savemat, loadmat

# === USER SETTINGS ===
fs = 48000  # Sampling rate
duration = 5.0  # seconds
fade_time = 0.1  # seconds for fade-in/out
frequencies = [125, 250, 500, 1000, 2000, 4000, 6000, 8000]  # Hz
target_spl = 60  # Target SPL in dB SPL
# amplitudes = [0.01]  # Signal amplitudes
ch = "R"  
# ch = "L"
RETSPL_file = "RETSPLdd45.mat"  # Load RETSPL table from .mat file
ATfreq = loadmat(RETSPL_file)['ATfreq'].squeeze()
AThresh = loadmat(RETSPL_file)['AThresh'].squeeze()
# mode = 'HL'
mode = 'SPL'  # 'HL' or 'SPL'
# retspl_table = { 
#     125: 45.0,
#     250: 27.0,
#     500: 13.5,
#     1000: 7.5,
#     2000: 9.0,
#     4000: 12.0,
#     6000: 16.0,
#     8000: 15.5
# }
# output_file = f"headphone_calibration_stereo_{ch}.csv"

# === Signal generator with fade-in/out ===
def generate_stereo_tone(f, amp):
    t = np.arange(0, duration, 1/fs)
    tone = amp * np.sin(2 * np.pi * f * t)
    fade_samples = int(fs * fade_time)
    fade_window = np.arange(0, 1, 1/fade_samples)
    tone[:fade_samples] *= fade_window
    tone[-fade_samples:] *= fade_window[::-1]


    return tone

# === Calibration loop ===
results = {}
results["headName"] = "DD-45"
results["Channel"] = ch
results["fx"] = []
results["Amplitude"] = []
results["SPL (dB SPL)"] = []
results["TrFuncAE"] = []
# rec = []

print("Calibration starting... Ensure SLM is ready and headphone is seated on artificial ear.")

# for f in frequencies:
#     for amp in amplitudes:    
#         print(f"\n Playing {f} Hz @ amp {amp:.3f} to {ch} channel...")
#         tone = generate_stereo_tone(f, amp)
#         print(tone.shape)
#         if ch == "L":
#             # rec = sd.playrec(tone, device=24, input_mapping=5, output_mapping=5, samplerate=fs, blocking=True, blocksize=1024)
#             sd.play(tone, fs, device=24, mapping=5)
#         elif ch == "R":
#             # rec = sd.playrec(tone, device=24, input_mapping=5, output_mapping=6, samplerate=fs, blocking=True, blocksize=1024)
#             sd.play(tone, fs, device=24, mapping=6)
#         else:
#             print("Invalid channel specified. Skipping.")
#             continue
#         sd.wait()

#         spl = input(f"Enter SPL (dB SPL) from SLM for {ch} channel at {f} Hz & amp {amp}: ")
#         spl = float(spl)
#         retspl = retspl_table[f]
#         hl = spl - retspl
#         print(f"HL: {hl} dB HL (SPL: {spl}, RETSPL: {retspl})")
#         TrFuncAE = 10 ** (spl / 20) * np.sqrt(2) * 2e-5 / amp
#         try:
#             results.append({
#                 "fx": f,
#                 "Amplitude": amp,
#                 "SPL (dB SPL)": spl,
#                 "TrFuncAE": TrFuncAE
#             })
#         except ValueError:
#             print("Invalid input. Skipping.")
#             continue
for f in frequencies:
    try:
        while True:
            amp = input("\nEnter amplitude: ")
            amp = float(amp)
            print(f"\n Playing {f} Hz @ amp {amp:.3f} to {ch} channel...")
            tone = generate_stereo_tone(f, amp)
            print(tone.shape)
            if ch == "L":
                sd.play(tone, fs, device=24, mapping=5)
            elif ch == "R":
                sd.play(tone, fs, device=24, mapping=8)
            else:
                print("Invalid channel specified. Skipping.")
                continue
            sd.wait()

            spl = input(f"Enter SPL (dB SPL) from SLM for {ch} channel at {f} Hz & amp {amp}: ")
            spl = float(spl)
            idx = np.where(ATfreq == f)
            retspl = AThresh[idx][0]
            hl = spl - retspl
            print(f"HL: {hl} dB HL (SPL: {spl}, RETSPL: {retspl})")
            TrFuncAE = 10 ** (spl / 20) * np.sqrt(2) * 2e-5 / amp
            if mode == 'HL':
                next_suggest = 10 ** ((target_spl - hl) / 20) * amp
                print(f"HL: {hl}, Amplitude: {amp}, Next suggested amplitude: {next_suggest:.3f}")
            elif mode == 'SPL':
                next_suggest = 10 ** ((target_spl - spl) / 20) * amp
                print(f"SPL: {spl}, Amplitude: {amp}, Next suggested amplitude: {next_suggest:.3f}")
            else:
                print("Invalid mode specified. Skipping next amplitude suggestion.")
                next_suggest = None
    except KeyboardInterrupt:
        results["fx"].append(f)
        results["Amplitude"].append(amp)
        results["SPL (dB SPL)"].append(spl)
        results["TrFuncAE"].append(TrFuncAE)
        print(f'{f} Hz data saved. Continuing to next frequency...')

#%% === Save to CSV ===
# df = pd.DataFrame(results)
# df.to_csv(output_file, index=False)
# print(f"\nCalibration data saved to {output_file}")
if ch == "L":
    output_file_mat = "DD-45left03-"+mode+".mat"
elif ch == "R":
    output_file_mat = "DD-45right03-"+mode+".mat"
else:
    print("Invalid channel specified. Skipping.")
savemat(output_file_mat, {'data': results})
print(f"Calibration data saved to {output_file_mat}")
#%% === Plot calibration curves ===
# plt.figure(figsize=(10, 6))
# for f in frequencies:
#     # subset = df[(df["Frequency (Hz)"] == f)]
#     # plt.plot(subset["Amplitude"], subset["" \
#     # "HL (dB HL)"], marker='o', label=f"{f} Hz ({ch})")
# plt.title("Calibration Curve: Amplitude → dB HL (L/R Channels)")
# plt.xlabel("Amplitude (Python)")
# plt.ylabel("dB HL")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("calibration_curve_stereo.png", dpi=300)
# plt.show()