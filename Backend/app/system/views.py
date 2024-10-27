from aiohttp.web_exceptions import (
    HTTPForbidden,
    HTTPUnauthorized,
    HTTPBadRequest,
    HTTPNotFound,
    HTTPServiceUnavailable,
)
from aiohttp_apispec import (
    request_schema,
)

from aiohttp_cors import CorsViewMixin
from app.web.mixin import AuthRequiredMixin
from app.web.app import View
from app.web.utils import json_response




class ReadStatView(AuthRequiredMixin, CorsViewMixin, View):
    async def get(self):
        read_books = await self.store.accessor.get_completed_books(self.request.user.id)
        return json_response(
            data={
                "read_amount": read_books,
            }
        )


class InterestView(AuthRequiredMixin, CorsViewMixin, View):
    async def get(self):
        interest_stat = await self.store.accessor.get_all_read_info_by_user_id(
            self.request.user.id
        )
        i = 0
        total_interest = 0
        for read_instance in interest_stat.data:
            for manga_instance in read_instance.read:
                i += 1
                total_interest += manga_instance.rating
        return json_response(
            data={
                "average_interest": total_interest / i,
            }
        )


class ReadTimeView(AuthRequiredMixin, CorsViewMixin, View):
    async def get(self):
        read_time_stat = await self.store.accessor.get_all_read_time_by_user_id(
            self.request.user.id
        )
        i = 0
        total_time = 0
        for read_instance in read_time_stat.data:
            i+=1
            total_time += (read_instance.start - read_instance.end).total_seconds()
        return json_response(
            data={
                "average_interest": int(total_time / i),
            }
        )


class ReadDaysView(AuthRequiredMixin, CorsViewMixin, View):
    async def get(self):
        read_periods = (
            await self.store.accessor.get_all_read_time_by_user_id_within_week(
                self.request.user.id
            )
        )
        return json_response(
            data={
                "periods": [
                    {"start": period.start, "end": period.end}
                    for period in read_periods.data
                ],
            }
        )


class MangaView( CorsViewMixin, View):
    async def get(self):
        try:
            manga_id = int(self.request.query.get("manga_id"))
        except:
            raise HTTPBadRequest(reason="Не указан идентификатор") 
        if not manga_id:
            raise HTTPBadRequest(reason="Не указан идентификатор")
        manga = await self.store.accessor.get_mangainfo(manga_id=manga_id)
        if manga is None:
            raise HTTPNotFound(resaon="Не существует указанной манги")
        if manga.theme:
            themes = await self.store.accessor.get_themes([theme.theme_id for theme in manga.theme])
            themes = themes.data
        else:
            themes= []
        if manga.ta:
            tas = await self.store.accessor.get_tas([ta.ta_id for ta in manga.ta])
            tas = tas.data
        else:
            tas= []
        if manga.author:
            authors = await self.store.accessor.get_authors([author.author_id for author in manga.author])
            authors = authors.data
        else:
            authors= []
        if manga.genre:
            genres = await self.store.accessor.get_genres([genre.genre_id for genre in manga.genre])
            genres = genres.data
        else:
            genres= []
        return json_response(
            data={
                "id": manga.id,
                "title": manga.title,
                "score": manga.score,
                "volumes": manga.volumes,
                "chapters": manga.chapters,
                "image": manga.image,
                "link": manga.link,
                "type": manga.type.name,
                "status": manga.status.name,
                "theme": [{"id":theme.id,"name":theme.name,}for theme in themes],
                "ta": [{"id":ta.id,"name":ta.name,}for ta in tas],
                "author": [{"id":author.id,"name":author.name,}for author in authors],
                "genre": [{"id":genre.id,"name":genre.name,}for genre in genres]
            }
        )
