// Manual equivalent of the GDS portion of analytics_worker.py.
// The worker drops an existing projection before running these statements.

CALL gds.graph.project(
  'aml_transaction_graph',
  'Account',
  {TRANSFER: {orientation: 'NATURAL'}}
)
YIELD graphName, nodeCount, relationshipCount;

// Degree supplies the existing Feast node_degree feature.
CALL gds.degree.write('aml_transaction_graph', {
  writeProperty: 'node_degree'
}) YIELD nodePropertiesWritten;

// SCC, rather than WCC, identifies directed circular money flows.
CALL gds.scc.write('aml_transaction_graph', {
  writeProperty: 'scc_community_id'
}) YIELD nodePropertiesWritten;

CALL gds.louvain.write('aml_transaction_graph', {
  writeProperty: 'louvain_community_id'
}) YIELD nodePropertiesWritten;

CALL gds.graph.drop('aml_transaction_graph') YIELD graphName;
