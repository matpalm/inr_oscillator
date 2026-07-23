rm /tmp/[ab]
jq . < runs/$1/opts.json > /tmp/a
jq . < runs/$2/opts.json > /tmp/b
sdiff /tmp/[ab]
