from .common_ui import render_spinn_page

def render_page():
    render_spinn_page(
        model_type="H3_separate",
        header='H3-PTFs: Need to know the soil texture and bulk density information (sand, silt, clay [×100%] & Bulk Density)',
        description='H3-separate: When training PTFs, first train the parameters of SWRC, and then train the parameters of HCC.',
        needs_bd=True
    )