# spheres123 notes

# github
git commit -am "nice"

git push -u origin master

git pull origin master

# to deploy
heroku login 

heroku plugins:install heroku-container-registry

heroku container:login

git clone https://github.com/heyredhat/spheres123

heroku create spheres123

heroku container:push web

heroku container:release web

# to config domain name
heroku domains:add www.xn--sphres-kva.com

heroku domains:add xn--sphres-kva.com
