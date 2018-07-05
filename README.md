# spheres123 notes

# to push to github
git push -u origin master

# github commit
git commit -am "nice"

# pull from github
git pull origin master

# to config domain name
heroku domains:add www.xn--sphres-kva.com

heroku domains:add xn--sphres-kva.com

# to deploy
heroku plugins:install heroku-container-registry

heroku container:login

git clone https://github.com/heyredhat/spheres123

heroku create spheres123

heroku container:push web

heroku container:release web

