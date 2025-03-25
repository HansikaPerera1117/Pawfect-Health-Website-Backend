import { Body, Controller, Param, Post, Query } from '@nestjs/common';
import { UserService } from './user.service';

@Controller('user')
export class UserController {
  constructor(private readonly userService: UserService) {}

  @Post('register')
  register(@Body() data: any) {
    return this.userService.register(data);
  }

  @Post('login')
  login(@Body() data: any) {
    return this.userService.loginUser(data.email, data.password);
  }

  remove(arg0: number) {
    return true;
  }
  
  update(arg0: number, arg1: { name: string }) {
    return true;
  }

  create(arg0: { name: string }) {
    return true;
  }

  findOne(arg0: number) {
    return true;
  }

  findAll() {
    return true;
  }

}
