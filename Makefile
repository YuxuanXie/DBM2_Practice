push:
	@git status; git stage .; git commit -m "Yx: $(yx)"; git push
