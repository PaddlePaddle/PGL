start_server(){
    ports=""
    for i in {7430..7439}; do
        nc -z localhost $i
        if [[ $? != 0 ]]; then
            ports="$ports $i"
        fi
    done
    python ./src/gen_redis_conf.py --ports $ports
    bash ./start_server.sh #启动服务器
}

start_server

