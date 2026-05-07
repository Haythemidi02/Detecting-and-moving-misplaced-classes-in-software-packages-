package com.example.config;

import org.springframework.web.bind.annotation.RestController;

@RestController
public class UserController {
    public String getUser() { return "ok"; }
}

