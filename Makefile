.PHONY: up down logs build

up:
	@touch bilibili_cookies.txt
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	@touch bilibili_cookies.txt
	docker compose up -d --build
