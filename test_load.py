from PySide6.QtWidgets import QApplication
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps

app = QApplication([])
vm = VMWorkspaceMaps()
vm._load_legacy_maps("examples/wafers.maps")
md = vm.store.get_map_data("wafer4_process1")
print("is_baseline_subtracted:", md.is_baseline_subtracted.any())
print("Y_baseline is not None:", md.Y_baseline is not None)
