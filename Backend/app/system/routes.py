from app.web.app import Application

from app.system.views import (
    ReadDaysView,
    ReadStatView,
    ReadTimeView,
    InterestView,
    MangaView,
)


def register_urls(application: Application):

    application.router.add_view("/user/read_amount", ReadStatView)
    application.router.add_view("/user/interest", InterestView)
    application.router.add_view("/user/read_time", ReadTimeView)
    application.router.add_view("/user/activity", ReadDaysView)
    application.router.add_view("/manga", MangaView)
