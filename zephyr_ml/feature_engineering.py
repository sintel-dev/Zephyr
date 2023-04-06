from sigpro import SigPro


def process_signals(es, signal_dataframe_name, signal_column, transformations, aggregations,
                    window_size, replace_dataframe=False, **kwargs):
    '''
    Process signals using SigPro.

    Apply SigPro transformations and aggregations on the specified entity from the
    given entityset. If ``replace_dataframe=True``, then the old entity will be updated.

    Args:
        es (featuretools.EntitySet):
            Entityset to extract signals from.
        signal_dataframe_name (str):
            Name of the dataframe in the entityset containing signal data to process.
        signal_column (str):
            Name of column or containing signal values to apply signal processing pipeline to.
        transformations (list[dict]):
            List of dictionaries containing the transformation primitives.
        aggregations (list[dict]):
            List of dictionaries containing the aggregation primitives.
        window_size (str):
            Size of the window to bin the signals over. e.g. ('1h).
        replace_dataframe (bool):
            If ``True``, will replace the entire signal dataframe in the EntitySet with the
            processed signals. Defaults to ``False``, creating a new child dataframe containing
            processed signals with the suffix ``_processed``.
    '''
    signal_df = es[signal_dataframe_name]
    time_index = signal_df.ww.time_index

    for relationship in es.relationships:
        child_name = relationship.child_dataframe.ww.name
        parent_name = relationship.parent_dataframe.ww.name

        if child_name == signal_df.ww.name and parent_name == 'turbines':
            old_relationship = relationship
            groupby_index = relationship.child_column.name

    pipeline = SigPro(transformations, aggregations, values_column_name=signal_column, **kwargs)

    processed_df, f_cols = pipeline.process_signal(
        signal_df,
        window=window_size,
        time_index=time_index,
        groupby_index=groupby_index,
        **kwargs
    )

    if replace_dataframe:
        es.add_dataframe(
            processed_df,
            signal_dataframe_name,
            time_index=time_index,
            index='_index')

    else:
        df_name = '{}_processed'.format(signal_df.ww.name)
        es.add_dataframe(processed_df, df_name, time_index=time_index, make_index=True,
                         index='_index')

        es.add_relationship('turbines', old_relationship.parent_column.name, df_name,
                            old_relationship.child_column.name)
