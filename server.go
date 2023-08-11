package main

import (
        "compress/gzip"
        "io"
        "log"
        "net/http"
        "os"
        "strings"
)

type gzipResponseWriter struct {
        io.Writer
        http.ResponseWriter
}

func (w gzipResponseWriter) Write(b []byte) (int, error) {
        return w.Writer.Write(b)
}

func wrapHandler(h http.Handler) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
                log.Println(r.URL.Path, r.RemoteAddr, r.UserAgent())
                if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
                        h.ServeHTTP(w, r)
                        return
                }
                w.Header().Set("Content-Encoding", "gzip")
                gz := gzip.NewWriter(w)
                defer gz.Close()
                gzw := gzipResponseWriter{Writer: gz, ResponseWriter: w}
                h.ServeHTTP(gzw, r)
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
        //log.Fatal(http.ListenAndServe(":8000", wrapHandler(http.FileServer(http.Dir("./static")))))
}