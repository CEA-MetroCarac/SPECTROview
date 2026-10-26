"""Unit tests for VSpectraList and SpectrumItemDelegate status indicator."""
from spectroview.view.components.v_spectra_list import VSpectraList, SpectrumItemDelegate


class TestSpectrumItemDelegate:
    def test_delegate_defaults(self, qapp):
        """Delegate initializes with circle shape and 8px size by default."""
        delegate = SpectrumItemDelegate()
        assert delegate.shape == "circle"
        assert delegate.size == 8
        assert delegate.margin_right == 10

    def test_list_set_indicator_shape(self, qapp):
        """VSpectraList allows switching indicator shape between circle, square, rounded_square."""
        lst = VSpectraList()
        assert isinstance(lst.itemDelegate(), SpectrumItemDelegate)
        assert lst.itemDelegate().shape == "circle"

        lst.set_indicator_shape("square")
        assert lst.itemDelegate().shape == "square"

        lst.set_indicator_shape("rounded_square")
        assert lst.itemDelegate().shape == "rounded_square"

    def test_render_items_with_and_without_status(self, qapp):
        """Delegate cleanly paints items with various status colors and transparent items."""
        lst = VSpectraList()
        sample_data = [
            {"fname": "Spectrum_Converged", "is_active": True, "has_fit": True, "fit_success": True},
            {"fname": "Spectrum_Unconverged", "is_active": True, "has_fit": True, "fit_success": False},
            {"fname": "Spectrum_Baselined", "is_active": True, "has_baseline": True},
            {"fname": "Spectrum_Cropped", "is_active": True, "is_cropped": True},
            {"fname": "Spectrum_Original", "is_active": True},
        ]
        lst.set_spectra_names(sample_data)
        assert lst.count() == 5

        # Check that background colors are set in model data
        item_converged = lst.item(0)
        assert item_converged.background().color().alpha() > 0

        item_orig = lst.item(4)
        assert item_orig.background().color().alpha() == 0

        # Verify painter rendering succeeds
        pix = lst.grab()
        assert not pix.isNull()
        assert pix.width() > 0
        assert pix.height() > 0
