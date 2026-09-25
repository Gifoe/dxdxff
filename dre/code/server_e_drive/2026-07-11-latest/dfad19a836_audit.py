from . import MODEL_VERSION
def model_audit(model):
 count=model.parameter_count();enabled=model.enabled_modules()
 if count>=250000:raise RuntimeError(f'TRACE-DRE model has {count} parameters, limit is <250000')
 if enabled.get('virtual_disconnection') and not enabled.get('true_ez_supernode'):raise RuntimeError('VDR enabled without true EZ supernode')
 return {'model_version':MODEL_VERSION,'parameter_count':count,'under_250k':True,'enabled_modules':enabled,'virtual_disconnection_definition':{'ez_node_source':'true_ez_channel_aggregation','nez_node_source':'top_q10_propagation_escape','ez_tau_source':'mean_true_ez_recruitment_time','max_nodes':13,'uses_full_channel_graph':False,'uses_gnn':False},'label_direction':{'NEZ':1,'EZ':0,'success':1,'failure':0}}
