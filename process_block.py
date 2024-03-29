import pdb

from IPython import embed
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import preprocessing

plt.ion()

def extract_targets_mvc(block_data, debug=False):
    """
    one target vector per block
    """
    # load signals
    sig_pulse = block_data['ADC_ 1.tev']
    ind_s = int(len(sig_pulse)*0.2)
    ind_e = int(len(sig_pulse)*0.8)
    if 'ADC_ 4.tev' in block_data.keys():
        if block_data['ADC_ 4.tev'][ind_s:ind_e].std() > \
            block_data['ADC_ 2.tev'][ind_s:ind_e].std():
            sig_mvc = block_data['ADC_ 4.tev']
        else:
            sig_mvc = block_data['ADC_ 2.tev']
    else:
        sig_mvc = block_data['ADC_ 2.tev']
    #sig_mvc = block_data['ADC_ 4.tev']

    # decimate them
    sig_pulse = sig_pulse[::10]
    sig_mvc = sig_mvc[::10]
    # preprocess them
    sig_pulse = np.abs(sig_pulse)
    sig_mvc = np.abs(sig_mvc)
    
    sig_pulse = scipy.signal.medfilt(sig_pulse, 501)
    sig_mvc = scipy.signal.medfilt(sig_mvc, 501)
    
    sig_pulse = (sig_pulse-sig_pulse.min()) / \
            (sig_pulse.max()-sig_pulse.min())
    
    
    # threshold
    sig_pulse[sig_pulse <0.5] = 0
    sig_pulse[sig_pulse >= 0.5] = 1
    
    step_ups = np.nonzero(np.diff(sig_pulse)>0)[0]
    step_downs = np.nonzero(np.diff(sig_pulse)<0)[0]
    
    try:
        step_diff = step_downs - step_ups
    except:
        return None
    
    # take major three ones
    step_ups = np.sort(step_ups[np.argsort(step_diff)[-3:]])
    step_downs = np.sort(step_downs[np.argsort(step_diff)[-3:]])
    
    # detrend mvc
    #sig_mvc = scipy.signal.detrend(sig_mvc)
    sig_mvc_bias = sig_mvc[:step_ups[0]].mean()
    sig_mvc = sig_mvc - sig_mvc_bias
    sig_mvc[sig_mvc<0] = 0.0
    
    if len(step_ups) != len(step_downs):
        if debug:
            plt.clf()
            plt.plot(sig_pulse)
            plt.plot(sig_mvc)
            #plt.plot(mvc_parts[0])
            #plt.plot(mvc_parts[1])
            #plt.plot(mvc_parts[2])
            plt.title('Failed')
            plt.draw()
            pdb.set_trace()
            #embed()
        return None
    # TODO: add signal quality check

    mvc_parts = []
    for  (step_up,step_down) in zip(step_ups,step_downs):
        mvc_parts.append(sig_mvc[step_up:step_down])
    
    extracted_targets = []
    forces = []
    areas = []
    if len(mvc_parts) != 3:
        return
    
    for part in mvc_parts:
        len_part = len(part)
        forces.append(np.median(np.sort(part)[int(0.95*len_part):]))
        areas.append(np.sum(part))
    
    #if np.sum(part) == 0:
    #    print('B1')
    #    pdb.set_trace()

    #areas = np.vstack(areas)
    #forces = np.vstack(forces)
    force = np.median(forces)
    area = np.median(areas)
    diff_ration = np.abs(1-(np.vstack(forces)/force))
    
    if (diff_ration > 0.25).any():
        print('WARNING: high MVC variation')
        if debug:
            plt.clf()
            plt.plot(sig_pulse)
            plt.plot(sig_mvc)
            #plt.plot(mvc_parts[0])
            #plt.plot(mvc_parts[1])
            #plt.plot(mvc_parts[2])
            plt.title('Failed')
            plt.draw()
            pdb.set_trace()
            #embed()
        #FIXME
        #return None
     
    #extracted_targets.append(force)
    #extracted_targets.append(area)
    extracted_targets = forces + areas
    #extracted_targets.append(areas)

   
    if debug:
        plt.clf()
        plt.plot(sig_pulse)
        plt.plot(sig_mvc)
        #plt.plot(mvc_parts[0])
        #plt.plot(mvc_parts[1])
        #plt.plot(mvc_parts[2])
        plt.title('accepted')
        plt.draw()
        pdb.set_trace()
        #embed()

    return extracted_targets
   
def extract_targets_emg(block_data, plot=False, debug=False):
    """
    one target vector per block
    """
    extracted_targets = []#{}
    # load signals
    sig_pulse = block_data['ADC_ 1.tev']
    ind_s = int(len(sig_pulse)*0.2)
    ind_e = int(len(sig_pulse)*0.8)
    sig_emg = block_data['sev'][2]
    # decimate them
    sig_pulse = sig_pulse[::10]
    sig_emg = sig_emg[::10]
    
    # preprocess them
    sig_emg -= sig_emg.mean()
    sig_pulse = np.abs(sig_pulse)
    sig_emg = np.abs(sig_emg)
     
    sig_pulse = scipy.signal.medfilt(sig_pulse, 501)
    #sig_emg = preprocessing.filter_iir_fwbw(sig_emg, fs_in=2000.0, 
    #        fs_out=2000.0, fc1=1.0, fc2=25.0, degree=3, plot=False)
    sig_emg = np.convolve(sig_emg, np.ones((400,))/400.0, 'valid')

    sig_pulse = (sig_pulse-sig_pulse.min()) / \
            (sig_pulse.max()-sig_pulse.min())
    
    # threshold
    sig_pulse[sig_pulse <0.5] = 0
    sig_pulse[sig_pulse >= 0.5] = 1
    
    if plot and False:
        plt.figure()
        plt.plot(sig_pulse[::1], 'k--', linewidth=1.5)
        plt.plot(sig_emg[::1]/sig_emg.max(), 'k', linewidth=1.0)
        plt.xlabel('Time (ms)', fontsize=16)
        plt.ylabel('Normalized Amplitude', fontsize=16)
        plt.legend(['Sync Signal', 'EMG Signal'], 
                loc='lower left')
        plt.ylim(0,1.1)
        plt.tight_layout()
        pdb.set_trace()
    
    step_ups = np.nonzero(np.diff(sig_pulse)>0)[0]
    step_downs = np.nonzero(np.diff(sig_pulse)<0)[0]
    
    try:
        step_diff = step_downs - step_ups
    except:
        return None
    step_ups = np.sort(step_ups[np.argsort(step_diff)[-3:]])
    step_downs = np.sort(step_downs[np.argsort(step_diff)[-3:]])
    if len(step_ups) != len(step_downs):
        if debug:
            plt.clf()
            plt.plot(sig_pulse)
            plt.plot(sig_emg)
            plt.title('Failed')
            plt.draw()
            pdb.set_trace()
            #embed()
        return None
    # TODO: add signal quality check

    emg_parts = []
    for  (step_up,step_down) in zip(step_ups,step_downs):
        emg_parts.append(sig_emg[step_up:step_down])
    
    if plot:
        plt.figure()
        part = emg_parts[0][::1] / emg_parts[0][::1].max()
        plt.plot(part, 'k')
        sel_point_y = np.median(np.sort(part)[int(0.95*len(part)):])
        max_point_y = np.max(part)
        range_x = np.arange(0,len(part))
        fig_1, = plt.plot(range_x,sel_point_y*np.ones((len(part),)), 
                'k--', linewidth=2.0)
        fig_2, = plt.plot(range_x,max_point_y*np.ones((len(part),)), 
                'k-.', linewidth=2.0)
        plt.xlabel('Time (ms)', fontsize=16)
        plt.ylabel('Normalized Amplitude', fontsize=16)
        plt.ylim(0,1.1)
        plt.legend([fig_1,fig_2],
                ['Proposed Method', 'Maximum Value'], loc='lower right')
        plt.tight_layout()
        pdb.set_trace()

    if len(emg_parts) != 3:
        if debug:
            plt.clf()
            plt.plot(sig_pulse)
            plt.plot(sig_emg)
            plt.title('Failed')
            plt.draw()
            pdb.set_trace()
            #embed()
        return None
    else:
        for part in emg_parts:
            extracted_targets.append(np.median(\
                np.sort(part)[int(0.95*len(part)):]))
        for part in emg_parts:
            extracted_targets.append(np.sum(part))
        #extracted_targets['emg_peak_1'] = np.median(\
        #        np.sort(emg_parts[0])[int(0.95*len(emg_parts[0])):])
        #extracted_targets['emg_peak_2'] = np.median(\
        #        np.sort(emg_parts[1])[int(0.95*len(emg_parts[1])):])
        #extracted_targets['emg_peak_3'] = np.median(\
        #        np.sort(emg_parts[2])[int(0.95*len(emg_parts[2])):])
    if debug:
        plt.clf()
        plt.plot(sig_pulse/10000.0)
        plt.plot(sig_emg)
        for part in emg_parts:
            plt.plot(part)
        plt.title('accepted')
        plt.draw()
        print(extracted_targets)
        pdb.set_trace()
        #embed()

    return extracted_targets
   
