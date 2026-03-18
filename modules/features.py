'''
Tools to extract features ECG/CXR/Waveform/Txt...
'''
import os
import zarr

class FeatureExtractor:
    '''
    Base class for all modalities
    '''
    def __init__(self, cohort_subjects:list[str], zarr_dir:str):
        self.cohort_subjects=cohort_subjects
        self.zarr_dir=zarr_dir
        os.makedirs(self.zarr_dir,exist_ok=True)

    def to_zarr(self):
        '''
        Perform the conversion raw-data-->the output zarr dataset
        '''
        raise NotImplementedError('Override in subclass')

    def filter_by_event(self, event_times):
        '''
        Optional: filter records by an event
        
        Args: 
            event_times (dict):{subject_id: np.datetime64(event_time)}
            
        '''
        pass

    def filter_by_event_time_window(self, event_times, window_before_minutes):
        '''
        Optional: filter records by a time window, ie include all records
        that occur within a certain time window before the event

        Args: 
            event_times (dict):{subject_id: np.datetime64(event_time)}
            window_before_minutes (int): minutes before event
        '''
        pass

    def filter_by_first(self, event_times):
        '''
        Optional: retrieve the first occurence of an event.
        Args: 
            event_times (dict):{subject_id: np.datetime64(event_time)}
            window_before_minutes (int): minutes before event
        '''
        pass
