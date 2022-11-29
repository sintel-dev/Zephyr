from sigpro import SigPro
import pandas as pd

def build_pipelines(transformations, aggregations, signal_columns, **kwargs):
    '''
    Args:
        es (ft.EntitySet):
        signal_dataframe (str): name of the dataframe containing signals data to process
        sigpro_pipeline: SigPro primitive pipeline to apply to signals data
        signal_columns (list[str]): column or list of columns containing signal values to apply signal processing pipeline to
        replace_datafframe (optional, Bool): if true, will replace the entire signal dataframe in the EntitySet with the
            processed signals. Defaults to false, creating a new child dataframe containing processed signals. 
    '''
    return [SigPro(transformations, aggregations, values_column_name=val, **kwargs) for val in signal_columns]


def process_signals(es, signal_dataframe_name, signal_column, transformations, aggregations, window_size, replace_dataframe=False, **kwargs):
    signal_df = es[signal_dataframe_name]
    time_index = signal_df.ww.time_index
    for relationship in es.relationships:
        if relationship.child_dataframe.ww.name == signal_df.ww.name and relationship.parent_dataframe.ww.name == 'turbines':
            old_relationship = relationship
            groupby_index = relationship.child_column.name
    
    pipeline = SigPro(transformations, aggregations, values_column_name=signal_column, **kwargs)

    processed_df, f_cols = pipeline.process_signal(signal_df, window=window_size, time_index=time_index, groupby_index=groupby_index, **kwargs)

    if replace_dataframe:
        es[signal_df] = processed_df
    else:
        df_name = '{}_processed'.format(signal_df.ww.name)
        es.add_dataframe(processed_df, df_name, time_index=time_index, make_index=True, index='_index')
        es.add_relationship('turbines', old_relationship.parent_column.name, df_name, old_relationship.child_column.name)
    return es
    


