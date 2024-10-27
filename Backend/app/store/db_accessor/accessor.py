import typing
from hashlib import sha256

from sqlalchemy import select, desc, update, delete, func
from sqlalchemy.orm import selectinload
import sqlalchemy.exc
from datetime import datetime, timedelta

from app.base.base_accessor import BaseAccessor
from app.system.dataclasses import (
    MangaInfoDC,
    ThemeDC,
    TADC,
    TypeDC,
    AuthorDC,
    MangaAuthorDC,
    ReadTimeDC,
    ReadTimeMangaDC,
    MangaTADC,
    MangaThemeDC,
    GenreDC,
    MangaGenreDC,
    MangaDC,
    ScoreDC,
    StatusDC,
    ReadTimeListDC,
    ReadTimeDetailsDC,
    ReadTimeMangaListDC,
    ReadTimeDetailsListDC,
    ThemeListDC,
    TAListDC,
    AuthorListDC,
    GenreListDC,
    ScoreListDC,
    ScoreFullDC,
    TGDC
)
from app.system.models import (
    TgModel,
    ThemeModel,
    TAModel,
    TypeModel,
    AuthorModel,
    MangaAuthorModel,
    ReadTimeModel,
    ReadTimeMangaModel,
    MangaTAModel,
    MangaThemeModel,
    GenreModel,
    MangaGenreModel,
    MangaModel,
    ScoreModel,
    StatusModel,
)
from app.user.models import UserModel
from app.user.dataclasses import UserDC, UserforRequest

if typing.TYPE_CHECKING:
    from app.web.app import Application


class DBAccessor(BaseAccessor):
    async def get_by_login(self, login: str) -> UserDC | None:
        async with self.app.database.session() as session:
            query = select(UserModel).where(UserModel.login == login)
            res = await session.scalars(query)
            user = res.one_or_none()
            if user:
                return UserDC(
                    id=user.id,
                    login=user.login,
                    name=user.name,
                    tg=user.tgid,
                    password=user.password,
                    registration_date=user.registration_date,
                    book_plan=user.book_plan,
                )
            return None

    async def get_by_tg(self, tg: int) -> UserDC | None:
        async with self.app.database.session() as session:
            query = select(UserModel).where(UserModel.tgid == tg)
            res = await session.scalars(query)
            user = res.one_or_none()
            if user:
                return UserDC(
                    id=user.id,
                    login=user.login,
                    name=user.name,
                    tg=user.tgid,
                    password=user.password,
                    registration_date=user.registration_date,
                    book_plan=user.book_plan,
                )
            return None

    async def get_by_id(self, id: int) -> UserDC | None:
        async with self.app.database.session() as session:
            query = select(UserModel).where(UserModel.id == id)
            res = await session.scalars(query)
            user = res.one_or_none()
            if user:
                return UserDC(
                    id=user.id,
                    login=user.login,
                    name=user.name,
                    tg=user.tgid,
                    password=user.password,
                    registration_date=user.registration_date,
                    book_plan=user.book_plan,
                )
            return None

    async def create_user(
        self, login: str, password: str, name: str
    ) -> UserforRequest | None:
        try:
            async with self.app.database.session() as session:
                user = UserModel(
                    login=login,
                    password=sha256(password.encode()).hexdigest(),
                    name=name,
                    registration_date=datetime.now(),
                )
                session.add(user)
                await session.commit()
                return UserforRequest(
                    id=user.id,
                    login=user.login,
                    name=user.name,
                )
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_tg_id(self, login: str, tgid: int) -> UserDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    update(UserModel).where(UserModel.login == login).values(tgid=tgid)
                )
                await session.execute(query)
                await session.commit()
                user = await self.get_by_login(login=login)
                return user
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_target(self, tgid: int, target: int) -> UserDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    update(UserModel)
                    .where(UserModel.tgid == tgid)
                    .values(book_plan=target)
                )
                await session.execute(query)
                await session.commit()
                user = await self.get_by_tg(tgid=tgid)
                return user
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_theme(self, name: str) -> ThemeDC | None:
        try:
            async with self.app.database.session() as session:
                theme = ThemeModel(
                    name=name,
                )
                session.add(theme)
                await session.commit()
                return theme.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_ta(self, name: str) -> TADC | None:
        try:
            async with self.app.database.session() as session:
                ta = TAModel(
                    name=name,
                )
                session.add(ta)
                await session.commit()
                return ta.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_author(self, name: str) -> AuthorDC | None:
        try:
            async with self.app.database.session() as session:
                author = AuthorModel(
                    name=name,
                )
                session.add(author)
                await session.commit()
                return author.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_genre(self, name: str) -> GenreDC | None:
        try:
            async with self.app.database.session() as session:
                genre = GenreModel(
                    name=name,
                )
                session.add(genre)
                await session.commit()
                return genre.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_type(self, name: str) -> TypeDC | None:
        try:
            async with self.app.database.session() as session:
                type_ = TypeModel(
                    name=name,
                )
                session.add(type_)
                await session.commit()
                return type_.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_status(self, name: str) -> StatusDC | None:
        try:
            async with self.app.database.session() as session:
                status = StatusModel(
                    name=name,
                )
                session.add(status)
                await session.commit()
                return status.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_manga(
        self,
        id:int,
        title: str,
        type_id: int,
        status_id: int,
        image: str,
        link: str,
        score: float | None = None,
        volumes: int | None = None,
        chapters: int | None = None,
    ) -> MangaDC | None:
        try:
            async with self.app.database.session() as session:
                manga = MangaModel(
                    id=id,
                    title=title,
                    type_id=type_id,
                    status_id=status_id,
                    image=image,
                    link=link,
                    score=score,
                    volumes=volumes,
                    chapters=chapters,
                )
                session.add(manga)
                await session.commit()
                return manga.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_mangatheme(self, manga_id: int, theme_id: int) -> MangaThemeDC | None:
        try:
            async with self.app.database.session() as session:
                mt = MangaThemeModel(manga_id=manga_id, theme_id=theme_id)
                session.add(mt)
                await session.commit()
                return mt.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_mangata(self, manga_id: int, ta_id: int) -> MangaTADC | None:
        try:
            async with self.app.database.session() as session:
                mt = MangaTAModel(manga_id=manga_id, ta_id=ta_id)
                session.add(mt)
                await session.commit()
                return mt.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_mangaauthor(
        self, manga_id: int, author_id: int, role: str
    ) -> MangaAuthorDC | None:
        try:
            async with self.app.database.session() as session:
                ma = MangaAuthorModel(manga_id=manga_id, author_id=author_id, role=role)
                session.add(ma)
                await session.commit()
                return ma.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_mangagenre(self, manga_id: int, genre_id: int) -> MangaGenreDC | None:
        try:
            async with self.app.database.session() as session:
                mg = MangaGenreModel(manga_id=manga_id, genre_id=genre_id)
                session.add(mg)
                await session.commit()
                return mg.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_score(
        self, manga_id: int, user_id: int, score: int
    ) -> ScoreDC | None:
        try:
            async with self.app.database.session() as session:
                score = ScoreModel(manga_id=manga_id, user_id=user_id, rating=score)
                session.add(score)
                await session.commit()
                return score.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_readtime(self, user_id: int) -> ReadTimeDC | None:
        try:
            async with self.app.database.session() as session:
                rt = ReadTimeModel(user_id=user_id, start=datetime.now())
                session.add(rt)
                await session.commit()
                return rt.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_readtime_end(self, id: int) -> None:
        try:
            async with self.app.database.session() as session:
                query = (
                    update(ReadTimeModel)
                    .where(ReadTimeModel.id == id)
                    .values(end=datetime.now())
                )
                await session.execute(query)
                await session.commit()
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_last_readtime(self, user_id: int) -> ReadTimeDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(ReadTimeModel)
                    .where(ReadTimeModel.user_id == user_id)
                    .order_by(ReadTimeModel.start.desc())
                    .limit(1)
                )
                res = await session.scalars(query)
                readtime = res.one_or_none()
                if readtime:
                    return ReadTimeDC(
                        id=readtime.id,
                        user_id=readtime.user_id,
                        start=readtime.start,
                        end=readtime.end,
                    )
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def del_readtime(self, id: int) -> None:
        try:
            async with self.app.database.session() as session:
                query = delete(ReadTimeModel).where(ReadTimeModel.id == id)
                await session.execute(query)
                await session.commit()
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_readtimemanga(
        self, read_id: int, manga_id: int, rating: int
    ) -> ReadTimeMangaDC | None:
        try:
            async with self.app.database.session() as session:
                rt = ReadTimeMangaModel(
                    readtime_id=read_id, manga_id=manga_id, rating=rating
                )
                session.add(rt)
                await session.commit()
                return rt.to_DC()
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def change_score(self, user_id: int, manga_id: int, rating: int) -> None:
        try:
            async with self.app.database.session() as session:
                query = (
                    update(ScoreModel)
                    .where(
                        (ScoreModel.user_id == user_id)
                        & (ScoreModel.manga_id == manga_id)
                    )
                    .values(rating=rating)
                )
                await session.execute(query)
                await session.commit()
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_last_30_readtime(self, user_id: int) -> ReadTimeListDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(ReadTimeModel)
                    .where(ReadTimeModel.user_id == user_id)
                    .order_by(ReadTimeModel.start.desc())
                    .limit(30)
                )
                res = await session.scalars(query)
                readtime = res.all()

                if readtime:
                    rl = [
                            ReadTimeDC(
                                id=readtime_instance.id,
                                user_id=readtime_instance.user_id,
                                start=readtime_instance.start,
                                end=readtime_instance.end,
                            )
                            for readtime_instance in readtime
                        ]
                    rl=rl[::-1]
                    return ReadTimeListDC(
                        data=rl
                    )
                else:
                    return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_completed_books(self, user_id: int) -> int | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(func.count(ScoreModel.rating))
                    .where(ScoreModel.user_id == user_id)
                    .group_by(ScoreModel.user_id)
                )
                res = await session.scalars(query)
                count = res.one_or_none()
                if count:
                    return count
                return 0
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_all_read_info_by_user_id(
        self, user_id: int
    ) -> ReadTimeDetailsListDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(ReadTimeModel)
                    .where(ReadTimeModel.user_id == user_id)
                    .options(selectinload(ReadTimeModel.readtimedetails))
                )
                res = await session.scalars(query)
                read_all = res.all()
                if read_all:
                    return ReadTimeDetailsListDC(
                        data=[
                            ReadTimeDetailsDC(
                                readtime=ReadTimeDC(
                                    id=read_instance.id,
                                    user_id=read_instance.user_id,
                                    start=read_instance.start,
                                    end=read_instance.end,
                                ),
                                read=[
                                    ReadTimeMangaDC(
                                        readtime_id=manga_instance.readtime_id,
                                        manga_id=manga_instance.manga_id,
                                        rating=manga_instance.rating,
                                    )
                                    for manga_instance in read_instance.readtimedetails
                                ],
                            )
                            for read_instance in read_all
                        ]
                    )
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_all_read_time_by_user_id(
        self,
        user_id: int,
    ) -> ReadTimeListDC | None:
        try:
            async with self.app.database.session() as session:
                query = select(ReadTimeModel).where(ReadTimeModel.user_id == user_id)
                res = await session.scalars(query)
                readtime = res.all()
                if readtime:
                    return ReadTimeListDC(
                        data=[
                            ReadTimeDC(
                                id=readtime_instance.id,
                                user_id=readtime_instance.user_id,
                                start=readtime_instance.start,
                                end=readtime_instance.end,
                            )
                            for readtime_instance in readtime
                        ]
                    )
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_all_read_time_by_user_id_within_week(
        self,
        user_id: int,
    ) -> ReadTimeListDC | None:
        try:
            async with self.app.database.session() as session:
                query = select(ReadTimeModel).where(
                    (ReadTimeModel.user_id == user_id)
                    & (ReadTimeModel.start > datetime.now() - timedelta(days=7))
                )
                res = await session.scalars(query)
                readtime = res.all()
                if readtime:
                    return ReadTimeListDC(
                        data=[
                            ReadTimeDC(
                                id=readtime_instance.id,
                                user_id=readtime_instance.user_id,
                                start=readtime_instance.start,
                                end=readtime_instance.end,
                            )
                            for readtime_instance in readtime
                        ]
                    )
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_mangainfo(self, manga_id: int) -> MangaInfoDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(MangaModel)
                    .where(MangaModel.id == manga_id)
                    .options(
                        selectinload(MangaModel.mangatype)).options(
                        selectinload(MangaModel.mangastatus)).options(
                        selectinload(MangaModel.mangatheme)).options(
                        selectinload(MangaModel.mangata)).options(
                        selectinload(MangaModel.mangaauthor)).options(
                        selectinload(MangaModel.mangagenre))
                )
                res = await session.scalars(query)
                manga = res.one_or_none()
                if manga:
                    return MangaInfoDC(
                        id=manga.id,
                        title=manga.title,
                        score=manga.score,
                        volumes=manga.volumes,
                        chapters=manga.chapters,
                        image=manga.image,
                        link=manga.link,
                        type=manga.mangatype.to_DC(),
                        status=manga.mangastatus.to_DC(),
                        theme=[theme.to_DC() for theme in manga.mangatheme],
                        ta=[ta.to_DC() for ta in manga.mangata],
                        author=[author.to_DC() for author in manga.mangaauthor],
                        genre=[genre.to_DC() for genre in manga.mangagenre],
                    )
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_themes(
        self,
        themes: list[int],
    ) -> ThemeListDC | None:
        try:
            async with self.app.database.session() as session:
                query = select(ThemeModel).filter(ThemeModel.id.in_(themes))
                res = await session.scalars(query)
                theme_res = res.all()
                if theme_res:
                    return ThemeListDC(data=[theme.to_DC() for theme in theme_res])
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_tas(self, tas: list[int]) -> TAListDC | None:
        try:
            async with self.app.database.session() as session:
                query = select(TAModel).filter(TAModel.id.in_(tas))
                res = await session.scalars(query)
                ta_res = res.all()
                if ta_res:
                    return TAListDC(data=[ta.to_DC() for ta in ta_res])
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_authors(self, authors: list[int]) -> AuthorListDC | None:
        try:
            async with self.app.database.session() as session:
                query = select(AuthorModel).filter(AuthorModel.id.in_(authors))
                res = await session.scalars(query)
                author_res = res.all()
                if author_res:
                    return ThemeListDC(data=[author.to_DC() for author in author_res])
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_genres(self, genres: list[int]) -> GenreListDC | None:
        try:
            async with self.app.database.session() as session:
                query = select(GenreModel).filter(GenreModel.id.in_(genres))
                res = await session.scalars(query)
                genre_res = res.all()
                if genre_res:
                    return ThemeListDC(data=[genre.to_DC() for genre in genre_res])
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def get_all_mangainfo(self) -> list[MangaInfoDC] | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(MangaModel)
                    .options(
                        selectinload(MangaModel.mangatype)).options(
                        selectinload(MangaModel.mangastatus)).options(
                        selectinload(MangaModel.mangatheme)).options(
                        selectinload(MangaModel.mangata)).options(
                        selectinload(MangaModel.mangaauthor)).options(
                        selectinload(MangaModel.mangagenre))
                    )
                
                res = await session.scalars(query)
                mangas = res.all()
                if mangas:
                    return [MangaInfoDC(
                        id=manga.id,
                        title=manga.title,
                        score=manga.score,
                        volumes=manga.volumes,
                        chapters=manga.chapters,
                        image=manga.image,
                        link=manga.link,
                        type=manga.mangatype.to_DC(),
                        status=manga.mangastatus.to_DC(),
                        theme=[theme.to_DC() for theme in manga.mangatheme],
                        ta=[ta.to_DC() for ta in manga.mangata],
                        author=[author.to_DC() for author in manga.mangaauthor],
                        genre=[genre.to_DC() for genre in manga.mangagenre],
                    )for manga in mangas]
                
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None
        
    async def get_user_scores(self, user_id:int)->ScoreListDC:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(ScoreModel).where(ScoreModel.user_id==user_id)
                    )
                res = await session.scalars(query)
                scores = res.all()
                if scores:
                    return ScoreListDC(data=[ScoreFullDC(
                        manga = await self.get_mangainfo(score.manga_id),
                        user_id=score.user_id,
                        rating=score.rating
                    ) for score in scores])             
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None
    
    async def add_chat(self, chatid: int) -> str | None:
        try:
            async with self.app.database.session() as session:
                chat = TgModel(chatid=chatid, state="INIT")
                session.add(chat)
                await session.commit()
                return "Yeap"
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None
    
    async def get_chat(self, chatid: int) -> TGDC|None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(TgModel)
                    .where(TgModel.chatid == chatid)
                )
                res = await session.scalars(query)
                chat = res.one_or_none()
                if chat:
                    return TGDC(userid=chat.userid,state=chat.state,chatid=chatid)
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None
    
    async def edit_chat_state(self, chatid: int,state:str) -> None:
        try:

            async with self.app.database.session() as session:
                query = (
                    update(TgModel)
                    .where(TgModel.chatid == chatid)
                    .values(state=state)
                )
                await session.execute(query)
                await session.commit()
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def add_chat_user(self, chatid: int,userid:int) -> None:
        try:
            async with self.app.database.session() as session:
                query = (
                    update(TgModel)
                    .where(TgModel.chatid == chatid)
                    .values(userid=userid)
                )
                await session.execute(query)
                await session.commit()
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None

    async def seacrh_manga_name(self, name: str) -> MangaDC | None:
        try:
            async with self.app.database.session() as session:
                query = (
                    select(MangaModel)
                    .filter(MangaModel.title.ilike(f'%{name}%')).limit(1)
                )
                res = await session.scalars(query)
                manga = res.one_or_none()
                if manga:
                    return manga.to_DC()
                return None
        except sqlalchemy.exc.IntegrityError:
            return None
        except sqlalchemy.exc.ProgrammingError:
            return None
    
