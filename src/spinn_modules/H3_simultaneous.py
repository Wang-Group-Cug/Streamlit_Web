from .common_ui import render_spinn_page

def render_page():
    render_spinn_page(
        model_type="H3_simultaneous",
        header='H3-PTFs: Need to know the soil texture and bulk density information (sand, silt, clay [×100%] & Bulk Density)',
        description='H3-simultaneous: When training PTF, consider both SWRC and HCC simultaneously.',
        needs_bd=True
    )