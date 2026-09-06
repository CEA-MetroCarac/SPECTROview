"""Discovery, validation, resources, and errors for the full desktop profile."""

import asyncio
import json

from mcp.shared.memory import create_connected_server_and_client_session

from spectroview.ai_agent.agent.ports import RecordingContext
from spectroview.ai_agent.mcp.server import create_desktop_mcp_server, create_mcp_server
from spectroview.application.errors import ApplicationAPIError


class DesktopContext(RecordingContext):
    def get_application_state(self):
        return {"ready": True, "active_workspace": "spectra"}

    def get_active_workspace(self):
        return {"name": "spectra", "index": 0}

    def get_current_selection(self):
        return {"workspace": "spectra", "spectrum_names": ["sample"]}

    def list_datasets(self, workspace=None):
        return [{"dataset_id": "spectra:sample", "workspace": "spectra"}]

    def get_dataset_info(self, dataset_id):
        return {"dataset_id": dataset_id, "point_count": 3}

    def get_spectrum(self, dataset_id, spectrum=None, processed=True, max_points=1000):
        if dataset_id == "spectra:missing":
            raise ApplicationAPIError("DATASET_NOT_FOUND", "Dataset is not loaded.")
        return {"dataset_id": dataset_id, "x": [1, 2, 3], "y": [4, 5, 6]}

    def get_current_spectrum(self, processed=True, max_points=1000):
        return self.get_spectrum("spectra:sample", None, processed, max_points)

    def get_fit_configuration(self, dataset_id=""):
        return {"dataset_id": dataset_id or "spectra:sample", "configuration": {}}

    def get_active_graph(self):
        raise ApplicationAPIError("NO_ACTIVE_GRAPH", "No graph is active.")

    def list_graph_configurations(self):
        return []

    def get_active_map(self):
        raise ApplicationAPIError("NO_ACTIVE_MAP", "No map is active.")


async def _catalog(full=True):
    server = (create_desktop_mcp_server(DesktopContext()) if full
              else create_mcp_server(DesktopContext()))
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        resources = (await session.list_resources()).resources
        return tools, resources


def _call(tool_name, arguments):
    async def run():
        server = create_desktop_mcp_server(DesktopContext())
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)
    return asyncio.run(run())


def _json_result(result):
    return json.loads(result.content[0].text)


class TestDiscoveryAndSchemas:
    def test_full_profile_exposes_high_value_capabilities(self):
        tools, resources = asyncio.run(_catalog())
        names = {tool.name for tool in tools}
        assert {
            "get_application_state", "list_datasets", "get_spectrum",
            "crop_spectrum", "fit_spectrum", "list_graphs", "export_results",
            "plot_graph", "update_graph",
        } <= names
        assert {str(resource.uri) for resource in resources} >= {
            "spectroview://application/state", "spectroview://workspace/current",
            "spectroview://datasets", "spectroview://graphs/current",
            "spectroview://fit/current",
        }

    def test_internal_chat_profile_stays_compact(self):
        tools, _ = asyncio.run(_catalog(full=False))
        assert {tool.name for tool in tools} == {
            "query_dataframe", "plot_graph", "get_statistics", "update_graph", "delete_graph"
        }

    def test_workspace_and_limits_are_typed(self):
        tools, _ = asyncio.run(_catalog())
        schemas = {tool.name: tool.inputSchema for tool in tools}
        assert set(schemas["fit_spectrum"]["properties"]["workspace"]["enum"]) == {
            "spectra", "maps"
        }
        assert schemas["get_spectrum"]["properties"]["max_points"]["minimum"] == 2
        assert schemas["get_spectrum"]["properties"]["max_points"]["maximum"] == 10000
        assert schemas["export_results"]["properties"]["overwrite"]["default"] is False

    def test_permission_hints_distinguish_reads_mutations_and_filesystem_writes(self):
        tools, _ = asyncio.run(_catalog())
        catalog = {tool.name: tool for tool in tools}
        assert catalog["get_application_state"].annotations.readOnlyHint is True
        assert catalog["crop_spectrum"].annotations.destructiveHint is True
        assert catalog["plot_graph"].annotations.readOnlyHint is False
        assert catalog["export_results"].annotations.openWorldHint is True
        assert catalog["delete_graph"].annotations.destructiveHint is True


class TestCallsAndErrors:
    def test_valid_call_returns_predictable_envelope(self):
        payload = _json_result(_call("get_application_state", {}))
        assert payload == {
            "ok": True,
            "result": {"ready": True, "active_workspace": "spectra"},
        }

    def test_domain_error_uses_stable_code(self):
        payload = _json_result(_call(
            "get_spectrum", {"dataset_id": "spectra:missing"}
        ))
        assert payload["ok"] is False
        assert payload["error"]["code"] == "DATASET_NOT_FOUND"

    def test_no_active_graph_is_a_structured_error(self):
        payload = _json_result(_call("get_active_graph", {}))
        assert payload["error"]["code"] == "NO_ACTIVE_GRAPH"

    def test_invalid_schema_arguments_are_rejected_by_mcp(self):
        result = _call("get_spectrum", {
            "dataset_id": "spectra:sample", "max_points": 1
        })
        assert result.isError is True

    def test_application_state_resource_is_readable(self):
        async def run():
            server = create_desktop_mcp_server(DesktopContext())
            async with create_connected_server_and_client_session(server._mcp_server) as session:
                await session.initialize()
                result = await session.read_resource("spectroview://application/state")
                return json.loads(result.contents[0].text)

        payload = asyncio.run(run())
        assert payload["ok"] is True
        assert payload["result"]["ready"] is True
