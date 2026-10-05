'''
Tools to extract features ECG/CXR/Waveform/Txt...
'''
import os

import pandas as pd
import zarr


def first_in_window(events:pd.DataFrame, anchors:pd.DataFrame, index:str, time_col:str, item_col:str,
                    value_col:str, hours)->pd.DataFrame:
    '''
    First non-missing value of each item within [hours[0], hours[1]] of each subject's anchor time,
    e.g. the first lab values within -12 h..+24 h of the index admission (same rule as the ECG modality window).
    Events are matched by subject and time only, not by admission id: values measured before the admission
    (e.g. in the emergency department, often without hadm_id) are kept if they fall in the window.
    events: long table with columns index, time_col, item_col, value_col (value_col numeric)
    anchors: one row per subject, columns index and anchor_time
    Returns a wide table: one row per subject (index), one column per item; subjects without values are absent.
    '''
    ev = events[[index, time_col, item_col, value_col]].dropna(subset=[value_col])
    ev = ev.merge(anchors[[index, "anchor_time"]], on=index)
    ev["_time"] = pd.to_datetime(ev[time_col], errors="coerce")
    offset_h = (ev["_time"] - ev["anchor_time"]).dt.total_seconds() / 3600
    ev = ev[offset_h.between(hours[0], hours[1])]
    first = ev.sort_values("_time").drop_duplicates([index, item_col])
    return first.pivot(index=index, columns=item_col, values=value_col)

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
