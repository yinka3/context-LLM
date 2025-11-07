import service_pb2
import service_pb2_grpc
from main import context, entity_resolve
from graph import graph_builder, graph_driver

driver = graph_driver.KnowGraph()
resolver = entity_resolve.EntityResolver()

builder = graph_builder.GraphBuilder(driver=driver)
context_manager = context.Context()

