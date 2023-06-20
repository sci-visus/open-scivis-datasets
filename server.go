package main

import (
        "log"
        "net/http"
        "os"
)

func wrapHandler(h http.Handler) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
                log.Println(r.URL.Path, r.RemoteAddr, r.UserAgent())
                h.ServeHTTP(w, r)
                // NOTE: we do not know if the request completed successfully (i.e., if the user did not cancel
                //      the download in middle)
        }
}

func main() {
        f, err := os.OpenFile("requests.log", os.O_RDWR | os.O_CREATE | os.O_APPEND, 0666)
        if err != nil {
                log.Fatal("Error opening log file")
        }
        defer f.Close()

        log.SetOutput(f)
        log.Fatal(http.ListenAndServeTLS(":443", "fullchain.pem", "privkey.pem", wrapHandler(http.FileServer(http.Dir("./static")))))
}