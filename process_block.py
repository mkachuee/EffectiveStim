import pdb

from IPython import embed
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


plt.ion()

def extract_targets(block_data, debug=False):
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
    sig_mvc = scipy.signal.detrend(sig_mvc)
    sig_mvc[sig_mvc<0] = 0.0
    
    # threshold
    sig_pulse[sig_pulse <0.5] = 0
    sig_pulse[sig_pulse >= 0.5] = 1
    
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
    for part in mvc_parts:
        len_part = len(part)
        forces.append(np.median(np.sort(part)[int(0.95*len_part):]))
        areas.append(np.sum(part))
    
    #if np.sum(part) == 0:
    #    print('B1')
    #    pdb.set_trace()

    areas = np.vstack(areas)
    forces = np.vstack(forces)
    force = np.median(forces)
    area = np.median(areas)
    diff_ration = np.abs(1-(forces/force))
    if (diff_ration > 0.5).any():
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
     
    #extracted_targets.append(force)
    #extracted_targets.append(area)
    extracted_targets.append(forces)
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

    return np.hstack(extracted_targets)
   
