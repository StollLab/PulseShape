# PusleShape

PulseShape is an EasySpin `pulse` function clone written in python. The major purpose for 
rewriting pulse in Python is to free the function from the proprietary MATLAB universe and 
make it easier to use on linux systems that often ship with e580 spectrometers. 

PulseShape is built around the `Pulse` object which accepts arguments similar to those 
used by the easyspin `pulse` function. 

## Example: sech\tanh pulse with resonator compensation
<table>
<tr>
<th>PulseShape</th>
<th>EasySpin</th>
</tr>
<tr>
<td>

```python
import numpy as np
import matplotlib.pyplot as plt
from PulseShape import Pulse

profile = np.loadtxt('data/Transferfunction.dat')
pulse = Pulse(pulse_time=0.150, 
              time_step=0.000625, 
              flip=np.pi, 
              freq=[40, 120], 
              type='sech/tanh', 
              beta=10, 
              profile=profile)

plt.figure(figsize=(5, 5))
plt.plot(pulse.time * 1000, pulse.IQ.real, label='real')
plt.plot(pulse.time * 1000, pulse.IQ.imag, label='imaginary')
plt.xlabel('time (ns)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
```
<img src="https://gitlab.com/mtessmer/PulseShape/-/raw/master/img/sechtanh.png" width="400"  class="center"/>


</td>
<td>

```matlab
Par = struct
Par.Type = 'sech/tanh';
Par.beta = 10;
Par.tp = 0.150;
Par.Phase = 0;
Par.Flip = pi;
Par.Frequency = [40 120]
Par.TimeStep=0.000625

filename = 'Transferfunction.dat';
delimiter = ' ';
formatSpec = '%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', ... 
                    delimiter, 'MultipleDelimsAsOne', ...
                    true, 'TextType', 'string');
fclose(fileID);

Par.FrequencyResponse = [dataArray{:, 1}, dataArray{:, 2}];

[t, IQ] = pulse(Par)
[t, IQ, modulation] = pulse(Par) 

figure(1)
hold on
plot(t, real(IQ))
plot(t, imag(IQ))
xlabel('time ns')
ylabel('Amplitude')
x0=10;
y0=10;
width=465;
height=448;
set(gcf,'position',[x0,y0,width,height])

```
<img src="https://gitlab.com/mtessmer/PulseShape/-/raw/master/img/sechtanhes.png" width="400" class="center"/>
</td>
</tr>
</table>

## Example: Working with multiple pulses

All `time`, `IQ`, other paramters and data are stored withing the `Pulse` object itself so it's easy to work with multiple pulses

```python
import numpy as np
import matplotlib.pyplot as plt
from PulseShape import Pulse

profile = np.loadtxt('data/Transferfunction.dat')
st_pulse = Pulse(pulse_time=0.150,
                 time_step=0.000625,
                 flip=np.pi,
                 freq=[40, 120],
                 type='sech/tanh',
                 beta=10,
                 profile=profile)

g_pulse = Pulse(pulse_time=0.06,
                time_step=0.000625,
                flip=np.pi,
                type='gaussian',
                trunc=0.1)

offsets = np.linspace(-20, 140, 256)
st_pulse.exciteprofile(offsets)
g_pulse.exciteprofile(offsets)

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 10))
ax1.set_title('Pulse IQ')
ax1.plot(st_pulse.time * 1e3, st_pulse.IQ.real, label=r'sech/tanh $\Re$', color='C0')
ax1.plot(st_pulse.time * 1e3, st_pulse.IQ.imag, label=r'sech/tanh $\Im$', alpha=0.5, color='C0')
ax1.plot(g_pulse.time * 1e3, g_pulse.IQ.real, label='gaussian', color='C1')
ax1.set_ylabel('Amplitude')
ax1.set_ylabel("Time (ns)")
ax1.legend()

ax2.set_title('Excitation Profile')
ax2.plot(offsets, st_pulse.Mz)
ax2.plot(offsets, g_pulse.Mz)
ax2.set_xlabel('Frequency Offset (MHz)')
ax2.set_ylabel('Mz')
plt.show()
```
<img src="https://gitlab.com/mtessmer/PulseShape/-/raw/master/img/g_st.png" width="800" class="center"/>
