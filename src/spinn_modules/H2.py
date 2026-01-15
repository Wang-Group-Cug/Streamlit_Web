from .common_ui import render_spinn_page

def render_page():
    render_spinn_page(
        model_type="H2",
        header='H2-PTFs: Need to know the soil texture information (sand, silt, clay [×100%])',
        description='',
        needs_bd=False
    )