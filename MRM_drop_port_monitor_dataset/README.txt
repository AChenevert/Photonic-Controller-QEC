SYNTHETIC FOUR-PORT MRM MONITOR DATASET
Gaussian and DRAG pulse trains, version 1, 2026-10-14

These data are generated for waveform/monitor algorithm testing. They are not
measured device data or a fitted model of a particular fabricated 4 K ring.
All numerical parameters and fault magnitudes are illustrative assumptions.

QUICK START
Use gaussian_train.csv or drag_train.csv. Each has 48,000 samples: 240 consecutive
20 ns slots, dt=0.1 ns, total duration 4.8 us. pulse_id starts at 1, sample_in_pulse
starts at 0. time_ns starts at 0 independently in each file. 20 ns is the slot
and pulse support, NOT the Gaussian FWHM. The untruncated FWHM is about 5.89 ns.
The peak input optical-power envelope is 120 uW. MRM commands vary between pulses.
Every 19th pulse is intentionally blocked at the thru port; this is not a fault.

Files ending instructor_truth.csv and pulse_labels_INSTRUCTOR.csv are simulation
ground truth for validation. Do not give these to an algorithm as monitor inputs.
The two train files contain known intended/reference waveforms but no fault labels.
parameters_and_validation.json records parameters and numerical checks.

COLUMN DEFINITIONS
time_ns: absolute time in this train.
pulse_id, sample_in_pulse, time_in_pulse_ns: segmentation and local time.
command_thru_transmission: desired optical POWER transmission T per pulse.
q_sign_command: known intended polarity command (+1, 0, -1); Gaussian has zero Q.
input/add/thru/drop_*_perfect_uW: intended input and nominal noiseless ring powers.
  The add port is not excited and add_*_perfect_uW is always zero.
*_measured_uW: detector readout expressed as calibrated equivalent optical power,
  including bandwidth, shot/electronic noise, gain/offset drift and quantization.
output_I_perfect_au, output_Q_perfect_au: ideal signed electrical envelopes after
  PD conversion/normalization, before carrier mixing. au means arbitrary units.
  These are NOT optical field amplitudes or calibrated voltages at a qubit.

Measured estimates can be slightly negative near zero because of receiver noise.
The actual optical powers in instructor_truth.csv are always nonnegative.
For Gaussian, Q is unused and all Q columns are zero (no Q detector simulated).
For DRAG, I and Q each have their own four-port ring lane. One ring does not
provide independent I and Q control. Q_abs columns are optical magnitudes.

PULSE AND RING MODEL
x=t_local-10 ns; sigma=2.5 ns; c=exp(-10^2/(2*sigma^2)).
g=max((exp(-x^2/(2*sigma^2))-c)/(1-c),0), truncated to the slot.
I=g; Q=beta*dg/dt, beta=0.5 ns, with derivative in units of inverse ns.
The DRAG sign convention is explicitly Q=+beta*dI/dt. It is illustrative and
must not be treated as a qubit-calibrated DRAG coefficient.
Optical input I=P0*g, Q_abs=P0*abs(Q), P0=120 uW.
Signed electrical Q is recovered using the independent sign/phase command.

The quasi-static single-mode add-drop ring uses FIELD decay rates r1=.50,
r2=.46, ri=.04 in nominal total-rate units, r=r1+r2+ri. delta is laser minus
resonance detuning in nominal HWHM units:
T=(delta^2+(ri+r2-r1)^2)/(delta^2+r^2)
D=4*r1*r2/(delta^2+r^2)
L=4*r1*ri/(delta^2+r^2)
T+D+L=1. Powers are input*T, input*D, input*L.
Nominal T=delta^2/(delta^2+1), D=.92/(delta^2+1), L=.08/(delta^2+1).
The command-to-detuning lookup is delta=sqrt(T_command/(1-T_command)).
Thru/drop faults are calculated from the same changed ring, not independently.
Loaded Q=15000 at 1550 nm gives linewidth ~12.9 GHz and a ~12 ps energy lifetime.
dt=100 ps does not resolve cavity transients. This is a quasi-static model and
does not simulate backscatter, resonance splitting, FSR, thermal feedback,
cryo voltage-to-detuning calibration, or microscopic carrier dynamics.
The slow-settling fault models the electrical bias response, not photon storage.

Photocurrent is responsivity*optical POWER (R=0.1 A/W), so the electrical envelope
scales with T, NOT sqrt(T). For any detector column, current in uA is 0.1*power_uW.
The synthetic receiver is a continuous 500 MHz single-pole lowpass with 0.318 ns
time constant; it retains state across slot boundaries. There are also small
fixed channel gain/offset mismatches, slow offset drift, variance-equivalent shot
noise and 0.12 uW stationary rms electronic noise. Noise is lowpass correlated
across 0.1 ns samples; the single-pole noise ENBW is pi/2 times 500 MHz. Exact
channel gains/offsets are recorded in parameters_and_validation.json.
Quantization is 150/(2^14-1) uW per step; readout range
is -2 to 150 uW, an illustrative input-referred range rather than an ADC voltage.

FAULT WINDOWS, PULSE IDS IN BOTH FILES
1-24 and 201-240: nominal with minor random bias/intensity/timing errors + noise.
25-48: progressive resonance drift, up to +0.18 nominal HWHM.
49-64: input optical power reduced to 85%.
65-80: intrinsic ring decay fraction increased from .04 to .12.
81-96: wrong bias, alternating +/-0.30 nominal HWHM.
97-112: 4 ns electrical bias settling from the previous command.
113-128: additional +/-0.45 ns input timing jitter.
129-144: Gaussian width +12%; DRAG Q-lane transmission 30% low.
145-160: only drop detector clipping (I at 30 uW, Q at 2.5 uW).
161-176: downstream LO phase error +15 degrees. Optical data cannot see it.
177-184: near-missing input pulse, input reduced to 5%.
185-200: combined input dip, detuning offset, extra loss and timing shift;
  DRAG additionally has Q-path programming error. See labels for exact values.

HOW TO TEST THE MONITOR
1. Extract peak, area, arrival time and width from measured input/drop powers.
2. Compare against the appropriate commanded perfect pulse, not a constant
   amplitude threshold. First compensate known receiver bandwidth/gain/offset.
   Filter the perfect reference with y[n]=alpha*y[n-1]+(1-alpha)*x[n],
   alpha=exp(-0.1/0.318309886), keeping state across the full train.
3. Predict thru from input+drop; use thru measurements only for evaluation of a
   drop-monitor predictor. With fixed nominal losses the exact optical relation
   is P_thru=P_input-(1+ri/r2)*P_drop. It is NOT generally P_input-P_drop.
4. Test drop-only separately from input+drop. Input and intrinsic-loss changes
   cannot in general be uniquely distinguished from drop-only power.
5. Validate pulse alarms and predicted waveform errors against instructor truth.
   A clipping event is a measurement fault, not necessarily a bad control pulse.
6. Report missed phase faults honestly. Upstream optical power cannot observe
   downstream LO phase/sign errors, mixer errors, or actual qubit syndromes.

The errors here are hardware/template errors, not observed Pauli X/Z events.
Mapping them to QEC decoder priors requires a separately justified qubit model
and identical syndromes/errors across any baseline vs monitor comparison.
This dataset does not demonstrate QEC improvement or hardware performance.

PYTHON EXAMPLE
import pandas as pd
df = pd.read_csv('gaussian_train.csv')
p = df[df.pulse_id == 50]
t = p.time_ns.to_numpy()
drop = p.drop_I_measured_uW.to_numpy()
thru_reference = p.thru_I_perfect_uW.to_numpy()

MATLAB EXAMPLE
D = readtable('drag_train.csv');
p = D(D.pulse_id == 50, :);
plot(p.time_ns, p.thru_I_measured_uW, p.time_ns, p.drop_I_measured_uW);
Q_measured_au = D.q_sign_command .* D.thru_Q_abs_measured_uW / 120;
% This is a reconstruction using the intended sign command, not measured LO phase.

REFERENCES
Dašić, INFOTEH 2013, coupled microring transfer/CMT models, equations 16-19.
https://infoteh.etf.ues.rs.ba/zbornik/2013/radovi/STS/STS-17.pdf
Motzoi et al., PRL 103, 110501 (2009), derivative quadrature for DRAG.
https://doi.org/10.1103/PhysRevLett.103.110501
